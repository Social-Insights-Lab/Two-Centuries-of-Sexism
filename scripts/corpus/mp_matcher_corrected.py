#!/usr/bin/env python3
"""
Corrected MP Matcher with accurate historical data
No hardcoded assumptions - all data verified from authoritative sources
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).resolve().parents[4]  # Up to hansard-nlp-explorer
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
import re
from typing import Optional, Tuple, List, Dict, Set
from collections import defaultdict
from datetime import datetime
import Levenshtein
import ast
from hansard.utils.path_config import Paths

class CorrectedMPMatcher:
    """MP Matcher with verified historical data"""

    def __init__(self, mp_data: pd.DataFrame = None):
        """Initialize with MP data"""
        if mp_data is None:
            mp_data = self._load_default_mp_data()

        self.mp_data = mp_data
        self._preprocess_data()
        self._build_indices()
        self._build_verified_title_database()
        self._build_ocr_corrections()

    def _load_default_mp_data(self) -> pd.DataFrame:
        """Load the default MP gender data (robust to CWD)."""
        candidates = [
            Paths.get_data_dir() / "house_members_gendered_updated.parquet",
            Path("src/hansard/data/house_members_gendered_updated.parquet"),
            Path("data/house_members_gendered_updated.parquet"),
        ]
        for path in candidates:
            if path.exists():
                return pd.read_parquet(path)
        raise FileNotFoundError("MP data not found in known locations: " + ", ".join(map(str, candidates)))

    def _preprocess_data(self):
        """Preprocess MP data for better matching"""
        # Add year columns
        self.mp_data['start_year'] = pd.to_datetime(
            self.mp_data['membership_start_date'], errors='coerce'
        ).dt.year
        self.mp_data['end_year'] = pd.to_datetime(
            self.mp_data['membership_end_date'], errors='coerce'
        ).dt.year

        # Extract chamber from organization
        self.mp_data['chamber'] = self.mp_data['organization_id'].apply(
            lambda x: 'Lords' if pd.notna(x) and 'lords' in str(x).lower() else 'Commons'
        )

        # Parse constituencies
        self.mp_data['constituency_list'] = self.mp_data['constituencies'].apply(
            self._parse_constituencies
        )

    def _parse_constituencies(self, const_data):
        """Parse constituency data"""
        if pd.isna(const_data):
            return []

        constituencies = []
        if isinstance(const_data, str):
            try:
                const_list = ast.literal_eval(const_data) if const_data.strip().startswith('[') else [const_data]
                for const in const_list:
                    if isinstance(const, dict) and 'constituency' in const:
                        constituencies.append(const['constituency'])
            except Exception:
                # Fallback: plain string constituency
                constituencies.append(const_data)
        elif isinstance(const_data, list):
            for const in const_data:
                if isinstance(const, dict) and 'constituency' in const:
                    constituencies.append(const['constituency'])

        return constituencies

    def _build_indices(self):
        """Build various lookup indices"""
        # Temporal + Chamber index
        self.temporal_chamber_index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Constituency index
        self.constituency_index = defaultdict(lambda: defaultdict(list))

        # Name variations index
        self.name_variations = defaultdict(set)

        for _, row in self.mp_data.iterrows():
            if pd.isna(row['person_name']):
                continue

            person = row['person_name']
            gender = row['gender_inferred']
            chamber = row['chamber']
            start_year = row['start_year']
            end_year = row['end_year']

            # Build name variations
            parts = person.split()
            if parts:
                surname = parts[-1].lower()
                # Add full name
                self.name_variations[surname].add(person)

                # Add initials version
                if len(parts) > 1:
                    initials = '.'.join([p[0] for p in parts[:-1]]) + f". {parts[-1]}"
                    self.name_variations[surname].add(initials)

            # Temporal + Chamber index
            if pd.notna(start_year) and pd.notna(end_year):
                for year in range(int(start_year), min(int(end_year) + 1, 2006)):
                    self.temporal_chamber_index[chamber][surname][year].append({
                        'full_name': person,
                        'gender': gender,
                        'row_idx': row.name,
                        'person_id': row['person_id']  # Add person_id to detect duplicates
                    })

            # Constituency index
            for const in row['constituency_list']:
                if const:
                    const_lower = const.lower()
                    if pd.notna(start_year) and pd.notna(end_year):
                        for year in range(int(start_year), min(int(end_year) + 1, 2006)):
                            self.constituency_index[const_lower][year].append({
                                'full_name': person,
                                'gender': gender
                            })

    def _build_verified_title_database(self):
        """
        Build database of titles to MPs
        Data source: UK Government official records (gov.uk/government/history/past-prime-ministers)
        Verified: 2024-09-20
        """
        self.title_database = {
            'prime minister': [
                # Date format: (start_date, end_date, name, gender)
                # Using exact dates from gov.uk
                ('1940-05-10', '1945-07-26', 'Winston Churchill', 'M'),
                ('1945-07-26', '1951-10-26', 'Clement Attlee', 'M'),
                ('1951-10-26', '1955-04-05', 'Winston Churchill', 'M'),
                ('1955-04-05', '1957-01-10', 'Anthony Eden', 'M'),
                ('1957-01-10', '1963-10-18', 'Harold Macmillan', 'M'),
                ('1963-10-18', '1964-10-16', 'Alec Douglas-Home', 'M'),
                ('1964-10-16', '1970-06-19', 'Harold Wilson', 'M'),
                ('1970-06-19', '1974-03-04', 'Edward Heath', 'M'),
                ('1974-03-04', '1976-04-05', 'Harold Wilson', 'M'),
                ('1976-04-05', '1979-05-04', 'James Callaghan', 'M'),
                ('1979-05-04', '1990-11-28', 'Margaret Thatcher', 'F'),
                ('1990-11-28', '1997-05-02', 'John Major', 'M'),
                ('1997-05-02', '2007-06-27', 'Tony Blair', 'M'),
            ],
            # Additional titles would be added here with verified dates
            # 'chancellor of the exchequer': [...],
            # 'foreign secretary': [...],
        }

    def _build_ocr_corrections(self):
        """Common OCR errors and corrections"""
        self.ocr_corrections = {
            # Common letter confusions
            'bavies': 'davies',
            'oavies': 'davies',
            'srnith': 'smith',
            'vvilliams': 'williams',
            "0'brien": "o'brien",  # Zero to O
            'b0nar': 'bonar',  # B0NAR LAW OCR error
            # Missing spaces
            'mrsmith': 'mr smith',
            'mrsthatcher': 'mrs thatcher',
            'mrdavies': 'mr davies',
        }

    def _extract_gender_from_title(self, speaker: str) -> Optional[str]:
        """
        Extract gender from title prefixes as fallback mechanism.

        Used when speaker cannot be matched to MP database but has
        a clear gender-indicating title prefix.

        Args:
            speaker: Original speaker string (may include title)

        Returns:
            'M', 'F', or None if no title prefix found

        Title mappings:
            Female: Mrs, Miss, Ms, Lady, Dame, Baroness
            Male: Mr, Sir, Lord, Colonel, Major, Captain, General, Admiral
        """
        if not speaker:
            return None

        speaker_lower = speaker.lower().strip()

        # Female titles (check first - more specific patterns)
        female_patterns = [
            r'^\s*(mrs|miss|ms|lady|dame|baroness)[\.\s]',
            r'\b(mrs|miss|ms|lady|dame|baroness)[\.\s]+[a-z]',  # Title in middle
        ]
        for pattern in female_patterns:
            if re.search(pattern, speaker_lower):
                return 'F'

        # Male titles
        male_patterns = [
            r'^\s*(mr|sir|lord|colonel|major|captain|general|admiral|viscount|earl|baron)[\.\s]',
            r'\b(mr|sir|lord)[\.\s]+[a-z]',  # Title in middle
        ]
        for pattern in male_patterns:
            if re.search(pattern, speaker_lower):
                return 'M'

        return None

    def _lookup_person_id(self, mp_name: str, date: str = None) -> Optional[str]:
        """
        Look up person_id for a matched MP name.

        Args:
            mp_name: The matched MP name (e.g., "Margaret Thatcher")
            date: Optional date to disambiguate if same name served at different times

        Returns:
            person_id or None
        """
        if not mp_name:
            return None

        # Search MP data for this name
        matches = self.mp_data[self.mp_data['person_name'] == mp_name]

        if len(matches) == 0:
            return None
        elif len(matches) == 1:
            return matches.iloc[0]['person_id']
        else:
            # Multiple matches - use date to disambiguate if provided
            if date:
                try:
                    check_date = pd.to_datetime(date)
                    for idx, row in matches.iterrows():
                        start = pd.to_datetime(row.get('start_year', '1800'), format='%Y', errors='coerce')
                        end = pd.to_datetime(row.get('end_year', '2100'), format='%Y', errors='coerce')
                        if pd.notna(start) and pd.notna(end):
                            if start <= check_date <= end:
                                return row['person_id']
                except:
                    pass

            # Fallback: return first match
            return matches.iloc[0]['person_id']

    def extract_consistent_gender_from_ambiguous(self, result: Dict) -> Dict:
        """
        For ambiguous matches, check if all candidates have same gender.
        If yes, assign gender with lower confidence (0.6) and mark as ambiguous_consistent_gender.

        Returns:
            dict: Updated result with gender if consistent, unchanged otherwise
        """
        if result['match_type'] != 'ambiguous':
            return result

        # Get all matches
        matches = result.get('matches', []) or result.get('possible_matches', [])
        if not matches:
            return result

        # Extract genders
        genders = set(m.get('gender') for m in matches if m.get('gender'))

        # If all matches have same gender
        if len(genders) == 1:
            gender = list(genders)[0]

            # Exclude known mixed-gender names (safeguard)
            speaker_lower = result.get('cleaned_speaker', '').lower()
            mixed_gender_names = ['lindsay', 'apsley', 'leslie wilson']
            if any(name in speaker_lower for name in mixed_gender_names):
                return result

            # Update result
            result['gender'] = gender
            result['match_type'] = 'ambiguous_consistent_gender'
            result['confidence'] = 0.6  # Lower than unique matches
            result['ambiguity_note'] = f'Cannot determine specific person, but all {len(matches)} candidates are {gender}'

        return result

    def match_comprehensive(self, speaker: str, date: str, chamber: str = None) -> Dict:
        """
        Comprehensive matching using all available strategies

        Returns:
            Dictionary with match results and confidence scores
        """
        results = {
            'original_speaker': speaker,
            'date': date,
            'chamber': chamber,
            'strategies_tried': [],
            'matches': [],
            'final_match': None,
            'confidence': 0.0,
            'match_type': 'no_match',
            'gender': None
        }

        # Clean speaker name
        speaker_clean = self._clean_speaker_name(speaker)
        results['cleaned_speaker'] = speaker_clean

        # Check if procedural
        if self._is_procedural(speaker_clean):
            results['match_type'] = 'procedural'
            return results

        # Try strategies in order of reliability

        # 0. Extract name from titled format: "The Minister of X (Name)"
        extracted_name = self._extract_titled_name(speaker)
        if extracted_name and extracted_name != speaker:
            # Recursively match the extracted name
            extracted_result = self.match_comprehensive(extracted_name, date, chamber)
            if extracted_result['final_match']:
                results['strategies_tried'].append('titled_extraction')
                results['matches'].append({
                    'mp_name': extracted_result['final_match'],
                    'person_id': extracted_result.get('person_id'),
                    'gender': extracted_result['gender'],
                    'confidence': extracted_result['confidence'] * 0.95,  # Slight discount for extraction
                    'method': 'titled_extraction'
                })
                results['final_match'] = extracted_result['final_match']
                results['person_id'] = extracted_result.get('person_id')
                results['confidence'] = extracted_result['confidence'] * 0.95
                results['match_type'] = 'titled_extraction'
                results['gender'] = extracted_result['gender']
                return results

        # 0b. Strip parenthetical suffix: "Mr. BONAR LAW (Leader of the House)" -> "Mr. BONAR LAW"
        stripped_name = self._strip_parenthetical_suffix(speaker)
        if stripped_name and stripped_name != speaker:
            # Recursively match the stripped name
            stripped_result = self.match_comprehensive(stripped_name, date, chamber)
            if stripped_result['final_match']:
                results['strategies_tried'].append('parenthetical_stripped')
                results['matches'].append({
                    'mp_name': stripped_result['final_match'],
                    'person_id': stripped_result.get('person_id'),
                    'gender': stripped_result['gender'],
                    'confidence': stripped_result['confidence'] * 0.95,  # Slight discount for stripping
                    'method': 'parenthetical_stripped'
                })
                results['final_match'] = stripped_result['final_match']
                results['person_id'] = stripped_result.get('person_id')
                results['confidence'] = stripped_result['confidence'] * 0.95
                results['match_type'] = 'parenthetical_stripped'
                results['gender'] = stripped_result['gender']
                return results

        # 1. Title resolution (highest confidence)
        title_match = self._resolve_title(speaker_clean, date)
        if title_match:
            results['strategies_tried'].append('title')
            results['matches'].append(title_match)
            # Transition days: explicitly mark ambiguous
            if title_match.get('method') == 'title_resolution_transition':
                results['match_type'] = 'ambiguous'
                results['ambiguity_count'] = 2  # at least outgoing/incoming
                results['final_match'] = None
                results['confidence'] = title_match['confidence']
                results['gender'] = title_match['gender']
                # Try to extract consistent gender from ambiguous matches
                results = self.extract_consistent_gender_from_ambiguous(results)
                return results
            if title_match['confidence'] >= 0.95:
                results['final_match'] = title_match['mp_name']
                results['confidence'] = title_match['confidence']
                results['match_type'] = 'title'
                results['gender'] = title_match['gender']
                results['person_id'] = self._lookup_person_id(title_match['mp_name'], date)
                return results

        # 2. Constituency matching
        const_match = self._match_by_constituency(speaker_clean, date)
        if const_match:
            results['strategies_tried'].append('constituency')
            results['matches'].append(const_match)
            if const_match['confidence'] >= 0.90:
                results['final_match'] = const_match['mp_name']
                results['confidence'] = const_match['confidence']
                results['match_type'] = 'constituency'
                results['gender'] = const_match['gender']
                results['person_id'] = self._lookup_person_id(const_match['mp_name'], date)
                return results

        # 3. Temporal + Chamber matching
        temporal_matches = self._match_temporal_chamber(speaker_clean, date, chamber)
        if temporal_matches:
            results['strategies_tried'].append('temporal_chamber')
            results['matches'].extend(temporal_matches)

            if len(temporal_matches) == 1:
                match = temporal_matches[0]
                results['final_match'] = match['mp_name']
                results['confidence'] = match['confidence']
                results['match_type'] = 'temporal_unique'
                results['gender'] = match['gender']
                results['person_id'] = self._lookup_person_id(match['mp_name'], date)
                return results
            elif len(temporal_matches) > 1:
                # Multiple matches - ambiguous
                results['match_type'] = 'ambiguous'
                results['ambiguity_count'] = len(temporal_matches)
                results['possible_matches'] = temporal_matches
                # Try to extract consistent gender from ambiguous matches
                results = self.extract_consistent_gender_from_ambiguous(results)
                return results

        # 4. Conservative fuzzy matching (last resort)
        fuzzy_match = self._fuzzy_match_conservative(speaker_clean, date, chamber)
        if fuzzy_match:
            results['strategies_tried'].append('fuzzy')
            results['matches'].append(fuzzy_match)
            if fuzzy_match['confidence'] >= 0.6:
                results['final_match'] = fuzzy_match['mp_name']
                results['confidence'] = fuzzy_match['confidence']
                results['match_type'] = 'fuzzy'
                results['gender'] = fuzzy_match.get('gender')
                results['person_id'] = self._lookup_person_id(fuzzy_match['mp_name'], date)
                return results

        # 5. Title-based gender inference (fallback when no MP match found)
        # Only used when we couldn't match to an MP but speaker has a clear title
        if results['gender'] is None and results['match_type'] == 'no_match':
            title_gender = self._extract_gender_from_title(speaker)
            if title_gender:
                results['strategies_tried'].append('title_gender')
                results['gender'] = title_gender
                results['match_type'] = 'title_gender_inference'
                results['confidence'] = 0.3  # Low confidence - only title, no person match
                results['note'] = 'Gender inferred from title prefix only (no MP match)'

        return results

    def _extract_titled_name(self, speaker: str) -> str:
        """
        Extract name from titled format: "The Minister of X (Name)"
        Returns the name in parentheses if pattern matches, otherwise None
        """
        if not speaker or '(' not in speaker:
            return None

        # Pattern: ends with (Name)
        import re
        match = re.search(r'\(([^)]+)\)$', speaker)
        if match:
            extracted = match.group(1).strip()
            # Only extract if it looks like a person name
            # Title must be at START and followed by a capitalized name (not "of", "the", etc.)
            # This avoids matching roles like "Lord of the Treasury"
            title_pattern = r'^(Mr|Mrs|Ms|Miss|Sir|Dame|Lord|Lady|Dr|Baroness|Viscount|Earl|Colonel|Captain|Major|Commander|Lieut|Lieutenant)\.?\s+[A-Z]'
            if re.match(title_pattern, extracted):
                return extracted

        return None

    def _strip_parenthetical_suffix(self, speaker: str) -> str:
        """
        Strip trailing parenthetical content that is NOT a name.
        Handles patterns like:
        - "Mr. BONAR LAW (Leader of the House)" -> "Mr. BONAR LAW"
        - "Captain ELLIOT (by Private Notice)" -> "Captain ELLIOT"
        - "Dr. MACNAMARA (resuming)" -> "Dr. MACNAMARA"
        - "Commander EYRES-MONSELL (Treasurer of the Household)" -> "Commander EYRES-MONSELL"
        - "Lieut.-Colonel Sir ROBERT A. SANDERS (Lord of the Treasury)" -> "Lieut.-Colonel Sir ROBERT A. SANDERS"
        - "Mr. ASQUITH (by Private Notice)." -> "Mr. ASQUITH"

        Does NOT strip if parentheses contain a name (handled by _extract_titled_name).
        """
        if not speaker or '(' not in speaker:
            return None

        import re
        # Match parentheses at end, optionally followed by punctuation
        match = re.search(r'\(([^)]+)\)[.,;:]*$', speaker)
        if match:
            paren_content = match.group(1).strip()
            # If parentheses contain a name (title at start followed by capitalized name), don't strip
            # This avoids stripping "(Mr. Baldwin)" but allows stripping "(Lord of the Treasury)"
            title_pattern = r'^(Mr|Mrs|Ms|Miss|Sir|Dame|Lord|Lady|Dr|Baroness|Viscount|Earl|Colonel|Captain|Major|Commander|Lieut|Lieutenant)\.?\s+[A-Z]'
            if re.match(title_pattern, paren_content):
                return None
            # Strip the parenthetical suffix (and any trailing punctuation)
            stripped = re.sub(r'\s*\([^)]*\)[.,;:]*$', '', speaker).strip()
            if stripped and stripped != speaker:
                return stripped

        return None

    def _clean_speaker_name(self, speaker: str) -> str:
        """Clean and normalize speaker name"""
        if not speaker:
            return ""

        speaker_lower = speaker.lower().strip()

        # Strip leading special characters and question numbers (common formatting)
        # Handles: "*MR. GODDARD", "24. Colonel YATE"
        speaker_lower = re.sub(r'^[*†§#&\d]+\.?\s*', '', speaker_lower)

        # Strip trailing punctuation (common in early Hansard: "Mr. HANBURY,")
        speaker_lower = speaker_lower.rstrip('.,;:')

        # Normalize spaces around hyphens (e.g., "EYRES - MONSELL" -> "EYRES-MONSELL")
        speaker_lower = re.sub(r'\s*-\s*', '-', speaker_lower)

        # Apply OCR corrections
        for error, correction in self.ocr_corrections.items():
            if error in speaker_lower:
                speaker_lower = speaker_lower.replace(error, correction)

        # Normalize apostrophes and fix missing spaces after titles
        speaker_lower = speaker_lower.replace('\u2019', "'")  # smart quote to ascii apostrophe
        # FIX: Only add space when title+period directly followed by lowercase (no space)
        # Matches "mr.smith" but not "mrs." or "ms armstrong"
        speaker_lower = re.sub(r'\b(mr|dr|sir)\.([a-z])', r'\1. \2', speaker_lower)
        # Separate regex for multi-char titles to avoid breaking "mrs" → "mr s"
        speaker_lower = re.sub(r'\b(mrs|miss)\.([a-z])', r'\1. \2', speaker_lower)

        # Normalize O' prefix (e.g., o'brien -> o'brien)
        speaker_lower = re.sub(r"\bo['']\s*([a-z])", r"o'\1", speaker_lower)

        return speaker_lower

    def _is_procedural(self, speaker: str) -> bool:
        """Check if speaker is a procedural entry"""
        procedural_patterns = [
            'procedural', 'the speaker', 'the deputy speaker',
            'the chairman', 'the deputy chairman', 'the clerk',
            'madam speaker', 'mr. speaker', 'several members',
            'hon. members', 'members'
        ]

        speaker_lower = speaker.lower()
        for pattern in procedural_patterns:
            if pattern in speaker_lower:
                return True
        return False

    def _resolve_title(self, speaker: str, date: str) -> Optional[Dict]:
        """Resolve ministerial titles using verified dates"""
        speaker_lower = speaker.lower()

        for title, appointments in self.title_database.items():
            if title in speaker_lower:
                try:
                    check_date = pd.to_datetime(date)

                    for start_date_str, end_date_str, mp_name, gender in appointments:
                        start_date = pd.to_datetime(start_date_str)
                        end_date = pd.to_datetime(end_date_str)

                        if start_date <= check_date <= end_date:
                            # For exact transition dates, lower confidence
                            if check_date == start_date or check_date == end_date:
                                return {
                                    'mp_name': mp_name,
                                    'person_id': self._lookup_person_id(mp_name, date),
                                    'gender': gender,
                                    'confidence': 0.7,  # Lower confidence on transition day
                                    'method': 'title_resolution_transition',
                                    'title': title,
                                    'term': f'{start_date_str} to {end_date_str}',
                                    'note': 'Transition date - ambiguous'
                                }
                            else:
                                # High confidence for dates clearly within term
                                return {
                                    'mp_name': mp_name,
                                    'person_id': self._lookup_person_id(mp_name, date),
                                    'gender': gender,
                                    'confidence': 0.95,
                                    'method': 'title_resolution',
                                    'title': title,
                                    'term': f'{start_date_str} to {end_date_str}'
                                }
                except Exception as e:
                    print(f"Error resolving title for {speaker} on {date}: {e}")
        return None

    def _match_by_constituency(self, speaker: str, date: str) -> Optional[Dict]:
        """Match by constituency reference"""
        patterns = [
            r"(?:the\s+)?member\s+for\s+([\w\-\'\s]+)",
            r"(?:the\s+)?mp\s+for\s+([\w\-\'\s]+)",
            r"representative\s+for\s+([\w\-\'\s]+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, speaker.lower())
            if match:
                constituency = match.group(1).strip().rstrip('.,;:')
                try:
                    year = pd.to_datetime(date).year
                    if constituency in self.constituency_index:
                        candidates = self.constituency_index[constituency].get(year, [])
                        if len(candidates) == 1:
                            return {
                                'mp_name': candidates[0]['full_name'],
                                'person_id': candidates[0].get('person_id'),
                                'gender': candidates[0]['gender'],
                                'confidence': 0.95,
                                'method': 'constituency'
                            }
                except:
                    pass
        return None

    def _match_temporal_chamber(self, speaker: str, date: str, chamber: str = None) -> List[Dict]:
        """Match using temporal and chamber constraints"""
        if not chamber:
            chamber = 'Commons'  # Default

        # Extract surname
        parts = speaker.split()
        if not parts:
            return []

        # Remove titles
        titles = ['mr', 'mrs', 'miss', 'ms', 'dr', 'sir', 'lord', 'lady']
        clean_parts = [p for p in parts if p.replace('.', '').lower() not in titles]

        if not clean_parts:
            return []

        # Handle O' prefix and common particles by merging if needed
        surname = clean_parts[-1].lower()
        if len(clean_parts) >= 2 and clean_parts[-2].lower() in {"o'", "o’", "mac", "mc", "van", "von", "de", "del", "di"}:
            surname = (clean_parts[-2] + ' ' + clean_parts[-1]).lower().replace('\u2019', "'")

        try:
            year = pd.to_datetime(date).year
        except:
            return []

        # Look up in index
        if surname not in self.temporal_chamber_index[chamber]:
            return []

        candidates = self.temporal_chamber_index[chamber][surname].get(year, [])

        # Filter by initials if provided
        if len(clean_parts) > 1:
            initial = clean_parts[0][0].lower()
            filtered = []
            for candidate in candidates:
                mp_name_parts = candidate['full_name'].split()
                if mp_name_parts and mp_name_parts[0][0].lower() == initial:
                    filtered.append(candidate)
            if filtered:
                candidates = filtered

        # Return matches with confidence
        if len(candidates) == 0:
            return []
        elif len(candidates) == 1:
            return [{
                'mp_name': candidates[0]['full_name'],
                'person_id': candidates[0].get('person_id'),
                'gender': candidates[0]['gender'],
                'confidence': 0.85,
                'method': 'temporal_chamber_unique'
            }]
        else:
            # Multiple candidates - check if they're the same person (duplicate memberships)
            unique_person_ids = set(c.get('person_id') for c in candidates if c.get('person_id'))

            if len(unique_person_ids) == 1:
                # Same person, multiple membership records - treat as unique match!
                return [{
                    'mp_name': candidates[0]['full_name'],
                    'person_id': candidates[0].get('person_id'),
                    'gender': candidates[0]['gender'],
                    'confidence': 0.90,  # High confidence - same person
                    'method': 'temporal_chamber_same_person'
                }]
            else:
                # Different people - truly ambiguous
                return [
                    {
                        'mp_name': c['full_name'],
                        'person_id': c.get('person_id'),
                        'gender': c['gender'],
                        'confidence': 0.4 / len(candidates),
                        'method': 'temporal_chamber_ambiguous'
                    }
                    for c in candidates
                ]

    def _fuzzy_match_conservative(self, speaker: str, date: str, chamber: str = None) -> Optional[Dict]:
        """Conservative fuzzy matching with safeguards"""
        # Only attempt if we have a reasonable name
        parts = speaker.split()
        if len(parts) < 2:
            return None

        surname = parts[-1].lower()
        if not chamber:
            chamber = 'Commons'

        # Look for similar surnames with edit distance <= 1
        best_match = None
        best_distance = float('inf')

        # If initial is provided, use it to constrain candidates for distance==1
        initial = parts[0][0].lower()

        for known_surname in self.temporal_chamber_index.get(chamber, {}).keys():
            distance = Levenshtein.distance(surname, known_surname)
            if distance <= 1 and distance < best_distance:
                # Verify temporal validity
                try:
                    year = pd.to_datetime(date).year
                    candidates = self.temporal_chamber_index[chamber][known_surname].get(year, [])
                    if candidates:
                        # Guard: if we have an initial in the input, require match for distance==1
                        if distance == 1 and candidates:
                            c_name = candidates[0]['full_name']
                            if not c_name or c_name[0].lower() != initial:
                                continue
                        best_match = candidates[0]
                        best_distance = distance
                except:
                    pass

        if best_match and best_distance <= 1:
            # Low confidence for fuzzy matches
            confidence = 0.7 if best_distance == 0 else 0.5
            return {
                'mp_name': best_match['full_name'],
                'person_id': best_match.get('person_id'),
                'gender': best_match['gender'],
                'confidence': confidence,
                'method': f'fuzzy_distance_{best_distance}'
            }

        return None
