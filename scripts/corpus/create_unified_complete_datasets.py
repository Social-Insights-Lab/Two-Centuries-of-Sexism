#!/usr/bin/env python3
"""
Create Unified Complete Datasets

Extracts speech segments from ALL 1.2M debates in processed_complete,
enriches with gender data from gender_complete where available,
and creates unified datasets with consistent schemas.

Input:
    - data-hansard/processed_complete (1.2M debates, all have full_text)
    - data-hansard/gender_complete (546k debates, gender-matched subset)

Output:
    - data-hansard/derived_complete/speeches_complete/ (~4-5M speeches)
    - data-hansard/derived_complete/debates_complete/ (1.2M debates)

Features:
    - Parallel processing by year
    - Speech extraction for ALL debates (not just MP-matched)
    - Gender enrichment where available (gender=None otherwise)
    - Unified schema (easy filtering)
    - Checkpoint/resume support
"""

import argparse
import sys
import json
import re
import copy
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import pickle
import signal

# Add src to path for imports
project_root = Path(__file__).resolve().parents[2]  # Up to hansard-nlp-explorer
sys.path.insert(0, str(project_root / 'src'))

from hansard.utils.speech_extractor import (
    extract_speeches_from_html,
    load_html_from_file,
    extract_speeches_from_file,
    extract_speeches_from_raw_html,
    get_raw_html_path
)
from hansard.utils.path_config import Paths


def normalize_speaker_name(speaker_name):
    """
    Normalize speaker name for better deduplication.

    Matches the normalization used in MP data (aliases_norm).

    Examples:
        "Mr. Winston Churchill" -> "winston churchill"
        "Mr. W. Churchill" -> "w churchill"
        "Sir Winston Churchill" -> "winston churchill"
        "The Secretary at War (Mr. Churchill)" -> "churchill"
    """
    if not speaker_name or pd.isna(speaker_name):
        return ""

    name = str(speaker_name).strip()

    # Extract name from parenthetical if it's a role description
    # E.g., "The Secretary at War (Mr. Churchill)" -> "Mr. Churchill"
    paren_match = re.search(r'\(([^)]+)\)$', name)
    if paren_match:
        extracted = paren_match.group(1).strip()
        # If what's in parentheses looks like a name (has honorific), use that
        if any(re.match(r'\b(Mr|Mrs|Miss|Ms|Dr|Sir|Lady|Lord)\b', extracted, re.IGNORECASE) for _ in [1]):
            name = extracted
        else:
            # Otherwise remove parentheses
            name = re.sub(r'\([^)]*\)', '', name)

    # Fix missing spaces after honorifics (Mr.Name -> Mr. Name)
    name = re.sub(r'(Mr|Mrs|Miss|Ms|Dr|Sir|Lady|Lord)\.([A-Z])', r'\1. \2', name, flags=re.IGNORECASE)

    # Remove honorifics and titles
    honorifics = [
        r'\bMr\.?\s*', r'\bMrs\.?\s*', r'\bMiss\.?\s*', r'\bMs\.?\s*',
        r'\bDr\.?\s*', r'\bSir\.?\s*', r'\bLady\.?\s*', r'\bLord\.?\s*',
        r'\bDame\.?\s*', r'\bRev\.?\s*', r'\bReverend\.?\s*',
        r'\bThe\s+Right\s+Hon\.?\s*', r'\bThe\s+Hon\.?\s*',
        r'\bColonel\.?\s*', r'\bMajor\.?\s*', r'\bCaptain\.?\s*',
        r'\bGeneral\.?\s*', r'\bAdmiral\.?\s*',
        r'\bThe\s+Secretary\s+at\s+War\s*', r'\bThe\s+Chancellor\s*',
        r'\bThe\s+Prime\s+Minister\s*', r'\bThe\s+Earl\s+of\s+',
        r'\bThe\s+Viscount\s+', r'\bThe\s+Baron\s+',
        r'\bThe\s+Baroness\s+', r'\bThe\s+Duchess\s+of\s+',
        r'\bThe\s+Duke\s+of\s+', r'\bThe\s+Marquess\s+of\s+',
    ]

    for honorific in honorifics:
        name = re.sub(honorific, '', name, flags=re.IGNORECASE)

    # Remove trailing punctuation and "rose"/"said" suffix
    name = re.sub(r'\s+(rose|said)$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[.,;:]+$', '', name)

    # Normalize whitespace
    name = re.sub(r'\s+', ' ', name).strip()

    # Convert to lowercase
    name = name.lower()

    return name


def clean_speaker_name(speaker_name):
    """
    Clean speaker name for display (remove artifacts but keep capitalization).

    Used for speakers without person_id (unmatched/ambiguous).

    Examples:
        "*MR. WILLIAM JOHNSTON" -> "William Johnston"
        "THE SECRETARY (Mr. Churchill,)" -> "Churchill"
        "*THE LORD ADVOCATE (Mr. A. GRAHAM MURRAY," -> "A. Graham Murray"
    """
    if not speaker_name or pd.isna(speaker_name):
        return ""

    name = str(speaker_name).strip()

    # Remove leading asterisk
    name = name.lstrip('*').strip()

    # Extract name from parenthetical if it's a role description
    # E.g., "THE LORD ADVOCATE (Mr. A. GRAHAM MURRAY," -> "Mr. A. GRAHAM MURRAY"
    paren_match = re.search(r'\(([^)]+)', name)  # Note: might have unclosed paren
    if paren_match:
        extracted = paren_match.group(1).strip().rstrip(',')
        # If what's in parentheses looks like a name (has honorific), use that
        if re.match(r'\b(Mr|Mrs|Miss|Ms|Dr|Sir|Lady|Lord)\b', extracted, re.IGNORECASE):
            name = extracted

    # Remove remaining parentheses
    name = re.sub(r'[()]+', '', name)

    # Remove honorifics and titles
    honorifics = [
        r'\bMr\.?\s*', r'\bMrs\.?\s*', r'\bMiss\.?\s*', r'\bMs\.?\s*',
        r'\bDr\.?\s*', r'\bSir\.?\s*', r'\bLady\.?\s*', r'\bLord\.?\s*',
        r'\bDame\.?\s*', r'\bRev\.?\s*', r'\bReverend\.?\s*',
        r'\bThe\s+Right\s+Hon\.?\s*', r'\bThe\s+Hon\.?\s*',
        r'\bColonel\.?\s*', r'\bMajor\.?\s*', r'\bCaptain\.?\s*',
        r'\bGeneral\.?\s*', r'\bAdmiral\.?\s*',
    ]

    for honorific in honorifics:
        name = re.sub(honorific, '', name, flags=re.IGNORECASE)

    # Remove trailing punctuation and commas
    name = re.sub(r'[.,;:]+$', '', name)

    # Normalize whitespace
    name = re.sub(r'\s+', ' ', name).strip()

    # Title case for better display
    name = name.title()

    return name


def load_person_name_lookup(mnis_data_path):
    """Load person_id -> person_name mapping from MP database."""
    try:
        mnis_df = pd.read_parquet(mnis_data_path)
        # Create lookup: person_id -> person_name
        lookup = mnis_df[['person_id', 'person_name']].drop_duplicates(subset=['person_id'])
        return dict(zip(lookup['person_id'], lookup['person_name']))
    except Exception as e:
        print(f"Warning: Could not load person_name lookup: {e}")
        return {}


def process_year(args):
    """Process a single year (for parallel execution)."""
    year, processed_dir, gender_dir, mnis_data_path, hansard_base = args

    # Load person_name lookup for canonical names
    person_name_lookup = load_person_name_lookup(mnis_data_path)

    # Read processed_complete for this year
    content_file = processed_dir / 'content' / str(year) / f'debates_{year}.jsonl'
    metadata_file = processed_dir / 'metadata' / f'debates_{year}.parquet'

    if not content_file.exists() or not metadata_file.exists():
        return {'year': year, 'speeches': 0, 'debates': 0, 'error': 'Files not found'}

    # Load gender_complete for this year (if exists)
    gender_file = gender_dir / f'debates_{year}_enhanced.parquet'
    gender_data = {}

    if gender_file.exists():
        try:
            gender_df = pd.read_parquet(gender_file)
            # Index by file_path for quick lookup
            for idx, row in gender_df.iterrows():
                file_path = row.get('file_path', '')
                if file_path:
                    gender_data[file_path] = {
                        'speaker_details': row.get('speaker_details', []),
                        'debate_id': row.get('debate_id', ''),
                        'topic': row.get('topic', ''),
                        'speech_segments': row.get('speech_segments', []),  # Pre-extracted speeches
                    }
        except Exception as e:
            # If gender file is corrupted or can't load, continue without it
            pass

    # Process all debates
    all_speeches = []
    all_debates = []

    debate_count = 0
    speech_count = 0

    try:
        with open(content_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                debate_data = json.loads(line)
                metadata = debate_data.get('metadata', {})

                # Extract basic info
                file_path = debate_data.get('file_path', '')
                full_text = debate_data.get('full_text', '')
                speakers = metadata.get('speakers', [])

                # Get content_hash first
                content_hash = debate_data.get('content_hash', '')

                # Check if we have gender data for this debate (match by file_path)
                gender_info = gender_data.get(file_path, {})
                speaker_details = gender_info.get('speaker_details', [])

                # Use debate_id from gender_complete if available, otherwise generate from content_hash
                if gender_info and gender_info.get('debate_id'):
                    debate_id = gender_info['debate_id']
                else:
                    debate_id = content_hash[:16] if content_hash else f"debate_{year}_{debate_count}"

                # Extract speeches from raw HTML (accurate, structure-based)
                # This replaces the buggy regex-based extraction
                speech_segments = []
                if file_path:
                    raw_html_path = get_raw_html_path(file_path, hansard_base)
                    if raw_html_path.exists():
                        speech_segments = extract_speeches_from_raw_html(raw_html_path)

                # Build speaker maps
                speaker_gender_map = {}
                speaker_party_map = {}
                speaker_constituency_map = {}
                speaker_person_id_map = {}

                for detail in speaker_details:
                    if isinstance(detail, dict):
                        orig_name = detail.get('original_name', '')
                        if orig_name:
                            speaker_gender_map[orig_name] = detail.get('gender')
                            speaker_party_map[orig_name] = detail.get('party')
                            speaker_constituency_map[orig_name] = detail.get('constituency')
                            speaker_person_id_map[orig_name] = detail.get('person_id')

                # Create speech records
                for seq_num, segment in enumerate(speech_segments, 1):
                    speaker = segment['speaker']

                    # Get MP data (None if not matched)
                    gender = speaker_gender_map.get(speaker)
                    matched_mp = gender is not None
                    party = speaker_party_map.get(speaker)
                    constituency = speaker_constituency_map.get(speaker)
                    person_id = speaker_person_id_map.get(speaker)

                    # Create canonical_name
                    # If we have person_id, use clean name from database
                    # Otherwise, apply basic cleaning to remove artifacts
                    if person_id and person_id in person_name_lookup:
                        canonical_name = person_name_lookup[person_id]
                    else:
                        canonical_name = clean_speaker_name(speaker)

                    speech = {
                        'speech_id': f"{debate_id}_speech_{seq_num}",
                        'debate_id': debate_id,
                        'file_path': file_path,
                        'sequence_number': seq_num,
                        'speaker': speaker,
                        'normalized_speaker': normalize_speaker_name(speaker),
                        'canonical_name': canonical_name,
                        'person_id': person_id,
                        'gender': gender,
                        'matched_mp': matched_mp,
                        'party': party,
                        'constituency': constituency,
                        # Handle both old format (position) and new format (contribution_id)
                        'contribution_id': segment.get('contribution_id', segment.get('position', seq_num)),
                        'is_question': segment.get('is_question', False),
                        'question_number': segment.get('question_number'),
                        'text': segment['text'],
                        'word_count': len(segment['text'].split()),
                        'year': year,
                        'decade': (year // 10) * 10,
                        'month': metadata.get('month', ''),
                        'date': metadata.get('reference_date', ''),
                        'chamber': metadata.get('chamber', ''),
                        'title': metadata.get('title', ''),
                        'topic': gender_info.get('topic', metadata.get('debate_topic', '')),
                        'hansard_reference': metadata.get('hansard_reference', ''),
                        'reference_volume': metadata.get('reference_volume', ''),
                        'reference_columns': metadata.get('reference_columns', ''),
                    }

                    all_speeches.append(speech)
                    speech_count += 1

                # Create debate record
                # Extract speakers that actually appear in speech_segments (most accurate)
                # This ensures speaker_genders only contains speakers who actually spoke
                speakers_from_segments = set()
                for segment in speech_segments:
                    if isinstance(segment, dict):
                        speaker = segment.get('speaker', '')
                        if speaker and str(speaker).strip():
                            speakers_from_segments.add(speaker.strip())
                
                # Also include speakers from metadata for completeness
                # But filter to only valid, non-empty names
                valid_speakers = []
                if speakers is not None:
                    # Convert numpy array to list if needed
                    if isinstance(speakers, np.ndarray):
                        speakers_list = speakers.tolist() if speakers.size > 0 else []
                    elif isinstance(speakers, list):
                        speakers_list = speakers
                    else:
                        speakers_list = list(speakers) if hasattr(speakers, '__iter__') else []
                    
                    # Filter: remove empty strings, None values, whitespace-only, and deduplicate
                    # Use sorted() for deterministic ordering
                    if speakers_list:
                        valid_speakers = sorted(set([
                            s.strip() for s in speakers_list
                            if s is not None and str(s).strip()
                        ]))
                
                # Combine: prefer speakers from segments, fallback to metadata
                all_speakers = list(speakers_from_segments) if speakers_from_segments else valid_speakers

                # Create normalized speaker list for unique speaker tracking
                # Use sorted() for deterministic ordering
                normalized_speakers = sorted(set([
                    normalize_speaker_name(speaker)
                    for speaker in all_speakers
                    if normalize_speaker_name(speaker)
                ]))

                # Create list of unique person_ids for matched MPs
                # Use sorted() for deterministic ordering
                unique_person_ids = sorted(set([
                    speaker_person_id_map.get(speaker)
                    for speaker in all_speakers
                    if speaker_person_id_map.get(speaker) is not None
                ]))

                # Create canonical_names list matching the speakers list
                canonical_names = []
                for speaker in all_speakers:
                    person_id = speaker_person_id_map.get(speaker)
                    if person_id and person_id in person_name_lookup:
                        canonical_names.append(person_name_lookup[person_id])
                    else:
                        canonical_names.append(clean_speaker_name(speaker))

                # CRITICAL: Build speaker_genders ONLY from speakers with gender matches
                # This prevents storing hundreds of None entries for unmatched speakers
                # Use .copy() to avoid Pandas sharing dict references across rows
                speaker_genders = {
                    str(speaker): str(speaker_gender_map.get(speaker))
                    for speaker in all_speakers
                    if speaker_gender_map.get(speaker) is not None
                }

                # Count statistics (including unmatched speakers)
                confirmed_mps = len(speaker_genders)
                unique_mp_count = len(unique_person_ids)
                female_mps = sum(1 for g in speaker_genders.values() if g == 'F')
                male_mps = sum(1 for g in speaker_genders.values() if g == 'M')

                debate = {
                    'debate_id': debate_id,
                    'file_path': file_path,
                    'content_hash': content_hash,
                    'year': year,
                    'decade': (year // 10) * 10,
                    'month': metadata.get('month', ''),
                    'date': metadata.get('reference_date', ''),
                    'chamber': metadata.get('chamber', ''),
                    'title': metadata.get('title', ''),
                    'topic': gender_info.get('topic', metadata.get('debate_topic', '')),
                    'hansard_reference': metadata.get('hansard_reference', ''),
                    'reference_volume': metadata.get('reference_volume', ''),
                    'reference_columns': metadata.get('reference_columns', ''),
                    'full_text': full_text,
                    'word_count': metadata.get('word_count', 0),
                    'speech_count': len(speech_segments),
                    'total_speakers': len(all_speakers),
                    'unique_normalized_speakers': len(normalized_speakers),
                    'unique_mp_count': unique_mp_count,  # Count of unique person_ids
                    'speakers': list(all_speakers),  # Original speaker names
                    'normalized_speakers': list(normalized_speakers),  # For unique speaker tracking
                    'canonical_names': list(canonical_names),  # Clean display names
                    'unique_person_ids': list(unique_person_ids),  # List of unique person_ids who spoke
                    'speaker_genders': json.dumps(speaker_genders),  # FIX: Store as JSON to avoid Parquet corruption
                    'confirmed_mps': confirmed_mps,
                    'female_mps': female_mps,
                    'male_mps': male_mps,
                    'has_female': female_mps > 0,
                    'has_male': male_mps > 0,
                    'gender_ratio': female_mps / (female_mps + male_mps) if (female_mps + male_mps) > 0 else None,
                }

                all_debates.append(debate)
                debate_count += 1

        return {
            'year': year,
            'speeches': speech_count,
            'debates': debate_count,
            'speeches_data': all_speeches,
            'debates_data': all_debates
        }

    except Exception as e:
        return {'year': year, 'speeches': 0, 'debates': 0, 'error': str(e)}


class UnifiedDatasetCreator:
    def __init__(self, processed_dir, gender_dir, output_dir, workers=12):
        self.processed_dir = Path(processed_dir)
        self.gender_dir = Path(gender_dir)
        self.output_dir = Path(output_dir)
        self.workers = workers

        # Path to raw HTML files for HTML-based speech extraction
        self.hansard_base = Paths.get_data_dir() / 'hansard'

        self.speeches_dir = self.output_dir / 'speeches_complete'
        self.debates_dir = self.output_dir / 'debates_complete'

        self.speeches_dir.mkdir(parents=True, exist_ok=True)
        self.debates_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.output_dir / 'unified_checkpoint.json'
        self.processed_years = set()

        # Load checkpoint
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                checkpoint = json.load(f)
                self.processed_years = set(checkpoint.get('processed_years', []))

        # Signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        self.interrupted = False

    def signal_handler(self, signum, frame):
        """Handle interruption gracefully."""
        print("\n\nInterrupted! Saving checkpoint...")
        self.save_checkpoint()
        print("Checkpoint saved. Resume by running script again.")
        sys.exit(0)

    def save_checkpoint(self):
        """Save progress."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'processed_years': list(self.processed_years)
            }, f, indent=2)

    def get_years_to_process(self, start_year, end_year):
        """Get list of years to process."""
        available_years = []

        for year_dir in (self.processed_dir / 'content').iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                year = int(year_dir.name)
                if start_year <= year <= end_year:
                    available_years.append(year)

        available_years.sort()

        # Filter out already processed
        years_to_process = [y for y in available_years if y not in self.processed_years]

        return years_to_process, available_years

    def process_all_years(self, start_year=1803, end_year=2005):
        """Process all years with parallel execution."""
        print("="*70)
        print("UNIFIED COMPLETE DATASETS CREATION")
        print("="*70)
        print(f"Processed dir: {self.processed_dir}")
        print(f"Gender dir: {self.gender_dir}")
        print(f"Output dir: {self.output_dir}")
        print(f"Workers: {self.workers}")
        print()

        years_to_process, all_years = self.get_years_to_process(start_year, end_year)

        if not years_to_process:
            print("All years already processed!")
            return

        print(f"Total years: {len(all_years)}")
        print(f"Already processed: {len(self.processed_years)}")
        print(f"To process: {len(years_to_process)}")
        print(f"Year range: {years_to_process[0]}-{years_to_process[-1]}")
        print()

        # input("Press Enter to start processing...")
        print()

        start_time = datetime.now()
        total_speeches = 0
        total_debates = 0

        # Get MP database path for canonical names
        project_root = Path(__file__).resolve().parents[2]  # Up to hansard-nlp-explorer
        mnis_data_path = project_root / 'data-hansard' / 'house_members_gendered_updated.parquet'

        # Process years in parallel
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            # Prepare arguments
            args_list = [
                (year, self.processed_dir, self.gender_dir, mnis_data_path, self.hansard_base)
                for year in years_to_process
            ]

            # Process with progress bar
            with tqdm(total=len(years_to_process), desc="Processing years") as pbar:
                for result in executor.map(process_year, args_list):
                    year = result['year']

                    if 'error' in result:
                        print(f"\n✗ {year}: {result['error']}")
                        pbar.update(1)
                        continue

                    # Save year data
                    if result['speeches_data']:
                        speeches_df = pd.DataFrame(result['speeches_data'])
                        speeches_df.to_parquet(
                            self.speeches_dir / f'speeches_{year}.parquet',
                            index=False
                        )

                    if result['debates_data']:
                        debates_df = pd.DataFrame(result['debates_data'])
                        debates_df.to_parquet(
                            self.debates_dir / f'debates_{year}.parquet',
                            index=False
                        )

                    total_speeches += result['speeches']
                    total_debates += result['debates']

                    self.processed_years.add(year)

                    # Save checkpoint every 10 years
                    if len(self.processed_years) % 10 == 0:
                        self.save_checkpoint()

                    pbar.set_postfix({
                        'speeches': f'{total_speeches:,}',
                        'debates': f'{total_debates:,}'
                    })
                    pbar.update(1)

        # Final checkpoint
        self.save_checkpoint()

        elapsed = (datetime.now() - start_time).total_seconds() / 3600

        print()
        print("="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        print(f"Total speeches created: {total_speeches:,}")
        print(f"Total debates created: {total_debates:,}")
        print(f"Time: {elapsed:.1f} hours")
        print()
        print(f"Output saved to:")
        print(f"  Speeches: {self.speeches_dir}/")
        print(f"  Debates: {self.debates_dir}/")

        # Create summary stats
        self.create_summary_stats()

    def create_summary_stats(self):
        """Create summary statistics across all years."""
        print()
        print("Creating summary statistics...")

        # Aggregate stats from all year files
        total_speeches = 0
        speeches_with_gender = 0
        speeches_by_gender = {'M': 0, 'F': 0, None: 0}

        for year_file in self.speeches_dir.glob('speeches_*.parquet'):
            df = pd.read_parquet(year_file)
            total_speeches += len(df)
            speeches_with_gender += df['gender'].notna().sum()

            for gender in ['M', 'F', None]:
                speeches_by_gender[gender] += (df['gender'] == gender).sum() if gender else df['gender'].isna().sum()

        total_debates = sum(
            len(pd.read_parquet(f))
            for f in self.debates_dir.glob('debates_*.parquet')
        )

        summary = {
            'created': datetime.now().isoformat(),
            'speeches': {
                'total': int(total_speeches),
                'with_gender': int(speeches_with_gender),
                'male': int(speeches_by_gender['M']),
                'female': int(speeches_by_gender['F']),
                'unmatched': int(speeches_by_gender[None]),
                'gender_match_rate': float(speeches_with_gender / total_speeches) if total_speeches > 0 else 0
            },
            'debates': {
                'total': int(total_debates)
            }
        }

        with open(self.output_dir / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print()
        print("SUMMARY:")
        print(f"  Total speeches: {total_speeches:,}")
        print(f"  With gender: {speeches_with_gender:,} ({speeches_with_gender/total_speeches*100:.1f}%)")
        print(f"  Without gender: {speeches_by_gender[None]:,} ({speeches_by_gender[None]/total_speeches*100:.1f}%)")
        print(f"  Male: {speeches_by_gender['M']:,}")
        print(f"  Female: {speeches_by_gender['F']:,}")
        print()
        print(f"  Total debates: {total_debates:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Create unified complete datasets with consistent schemas'
    )
    parser.add_argument('--processed-dir', required=True,
                       help='Input processed_complete directory')
    parser.add_argument('--gender-dir', required=True,
                       help='Input gender_complete directory')
    parser.add_argument('--output-dir', default=str(Paths.DERIVED_DATA),
                       help='Output directory (default: versioned derived_v2)')
    parser.add_argument('--start-year', type=int, default=1803,
                       help='Start year')
    parser.add_argument('--end-year', type=int, default=2005,
                       help='End year')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: 75%% of CPU cores)')
    parser.add_argument('--reset', action='store_true',
                       help='Reset checkpoint and start fresh')

    args = parser.parse_args()

    if args.workers is None:
        args.workers = max(1, int(mp.cpu_count() * 0.75))

    # Handle reset
    if args.reset:
        checkpoint = Path(args.output_dir) / 'unified_checkpoint.json'
        if checkpoint.exists():
            checkpoint.unlink()
            print("Checkpoint reset.\n")

    # Create processor
    creator = UnifiedDatasetCreator(
        processed_dir=args.processed_dir,
        gender_dir=args.gender_dir,
        output_dir=args.output_dir,
        workers=args.workers
    )

    # Process
    creator.process_all_years(
        start_year=args.start_year,
        end_year=args.end_year
    )


if __name__ == "__main__":
    # Set multiprocessing start method for macOS
    mp.set_start_method('spawn', force=True)
    main()
