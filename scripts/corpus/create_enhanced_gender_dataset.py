#!/usr/bin/env python3
"""
Create ENHANCED gender analysis dataset with full debate text from existing extracted content
Processes all years of data (1803-2005) with checkpointing
Uses pre-extracted text from processed_fixed/content directory
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).resolve().parents[2]  # Up to hansard-nlp-explorer
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
from hansard.matching.mp_matcher_corrected import CorrectedMPMatcher
from hansard.utils.path_config import Paths
from hansard.utils.speech_extractor import (
    extract_speeches_from_html,
    load_html_from_file,
    extract_speeches_from_file,
    extract_speeches_from_raw_html,
    get_raw_html_path
)
import json
from tqdm import tqdm
import hashlib
from datetime import datetime
import pickle
import argparse
import signal
import os
import re
import gzip
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from bs4 import BeautifulSoup


# Module-level functions for parallel processing
# Note: extract_speeches_from_raw_html and get_raw_html_path are imported from
# hansard.utils.speech_extractor to avoid code duplication


def process_year_parallel(args):
    """Process a single year (for parallel execution)"""
    year, content_base, metadata_base, mnis_data_path, hansard_base = args

    # Load MNIS data and create matcher in worker process
    mnis_df = pd.read_parquet(mnis_data_path)
    matcher = CorrectedMPMatcher(mnis_df)

    # Load metadata for this year
    metadata_file = metadata_base / f"debates_{year}.parquet"
    if not metadata_file.exists():
        return {'year': year, 'debates': 0, 'error': 'Metadata file not found'}

    debates_df = pd.read_parquet(metadata_file)

    # Load texts for this year (used for full_text and metadata)
    year_content_file = content_base / str(year) / f"debates_{year}.jsonl"
    texts_by_path = {}

    if year_content_file.exists():
        try:
            with open(year_content_file, 'r') as f:
                for line in f:
                    debate_data = json.loads(line)
                    file_path = debate_data.get('file_path', '')
                    texts_by_path[file_path] = {
                        'full_text': debate_data.get('full_text', ''),
                        'content_hash': debate_data.get('content_hash', ''),
                        'extraction_timestamp': debate_data.get('extraction_timestamp', '')
                    }
        except Exception as e:
            return {'year': year, 'debates': 0, 'error': f'Failed to load texts: {e}'}

    # Process debates
    year_debates = []

    for idx, debate in debates_df.iterrows():
        speakers = debate.get('speakers', [])
        date = debate.get('reference_date', f'{year}-01-01')
        chamber = debate.get('chamber', 'Commons')

        # Convert to list if numpy array
        if isinstance(speakers, np.ndarray):
            speakers = speakers.tolist()

        if not isinstance(speakers, list) or len(speakers) == 0:
            continue

        # Match speakers
        matched_mps = []
        female_mps = []
        male_mps = []
        speaker_details = []
        ambiguous = 0
        unmatched = 0
        unmatched_names = []

        for speaker in speakers:
            if pd.notna(speaker):
                result = matcher.match_comprehensive(str(speaker), date, chamber)

                if result['match_type'] in ['temporal_unique', 'title', 'constituency', 'titled_extraction', 'temporal_chamber_same_person', 'ambiguous_consistent_gender']:
                    if result.get('confidence', 0) >= 0.6:  # Lower threshold to include ambiguous_consistent_gender (0.6)
                        mp_name = result.get('final_match')
                        matched_mps.append(mp_name)

                        speaker_info = {
                            'original_name': str(speaker),
                            'matched_name': mp_name,
                            'person_id': result.get('person_id'),
                            'gender': result.get('gender'),
                            'party': result.get('party'),
                            'constituency': result.get('constituency'),
                            'confidence': result.get('confidence'),
                            'match_type': result['match_type']
                        }
                        speaker_details.append(speaker_info)

                        if result.get('gender') == 'F':
                            female_mps.append(mp_name)
                        elif result.get('gender') == 'M':
                            male_mps.append(mp_name)
                elif result['match_type'] == 'ambiguous':
                    ambiguous += 1
                else:
                    unmatched += 1
                    unmatched_names.append(str(speaker))

        # Only keep debates with at least one confirmed MP
        if matched_mps:
            file_path = debate.get('file_path', '')
            debate_text_data = texts_by_path.get(file_path, {})
            full_text = debate_text_data.get('full_text', '')

            # Extract speech segments from raw HTML (accurate, structure-based)
            speech_segments = []
            if file_path:
                raw_html_path = get_raw_html_path(file_path, hansard_base)
                if raw_html_path.exists():
                    speech_segments = extract_speeches_from_raw_html(raw_html_path)

            debate_id = hashlib.md5(f"{file_path}_{date}".encode()).hexdigest()[:16]
            decade = (year // 10) * 10

            debate_record = {
                'debate_id': debate_id,
                'year': year,
                'decade': decade,
                'month': debate.get('month'),
                'reference_date': date,
                'chamber': chamber,
                'title': debate.get('title'),
                'topic': debate.get('debate_topic'),
                'hansard_reference': debate.get('hansard_reference'),
                'reference_volume': debate.get('reference_volume'),
                'reference_columns': debate.get('reference_columns'),
                'debate_text': full_text,
                'text_length': len(full_text) if full_text else 0,
                'has_text': bool(full_text),
                'content_hash': debate_text_data.get('content_hash'),
                'extraction_timestamp': debate_text_data.get('extraction_timestamp'),
                'speech_segments': speech_segments,
                'speech_count': len(speech_segments),
                'total_speakers': len(speakers),
                'confirmed_mps': len(matched_mps),
                'female_mps': len(female_mps),
                'male_mps': len(male_mps),
                'has_female': len(female_mps) > 0,
                'has_male': len(male_mps) > 0,
                'female_names': female_mps,
                'male_names': male_mps,
                'speaker_details': speaker_details,
                'ambiguous_speakers': ambiguous,
                'unmatched_speakers': unmatched,
                'unmatched_names': unmatched_names,
                'gender_ratio': len(female_mps) / (len(female_mps) + len(male_mps)) if (female_mps or male_mps) else None,
                'word_count': debate.get('word_count', 0),
                'line_count': debate.get('line_count', 0),
                'char_count': debate.get('char_count', 0),
                'file_path': file_path,
                'file_name': debate.get('file_name'),
                'file_size': debate.get('file_size', 0),
                'file_modified': debate.get('file_modified'),
                'processing_timestamp': datetime.now().isoformat(),
                'matcher_version': 'corrected_v1'
            }

            year_debates.append(debate_record)

    return {
        'year': year,
        'debates': len(year_debates),
        'total_debates_in_year': len(debates_df),
        'debates_data': year_debates
    }


class EnhancedGenderDatasetCreator:
    def __init__(self, output_dir=None, input_dir=None, checkpoint_file="checkpoint.pkl", workers=12):
        if output_dir is None:
            # Default to versioned output directory
            output_dir = Paths.GENDER_ENHANCED_DATA
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_dir / checkpoint_file
        self.workers = workers

        # Path to extracted content (configurable)
        if input_dir is None:
            # Use versioned input, fallback to v1 if it doesn't exist
            if Paths.PROCESSED_DATA.exists():
                input_dir = Paths.PROCESSED_DATA
            else:
                input_dir = Paths.PROCESSED_V1
        else:
            input_dir = Path(input_dir)

        self.content_base = input_dir / 'content'
        self.metadata_base = input_dir / 'metadata'

        # Path to raw HTML files for HTML-based speech extraction
        self.hansard_base = Paths.get_data_dir() / 'hansard'

        # Store path to MNIS data for worker processes
        self.mnis_data_path = Paths.get_data_dir() / "house_members_gendered_updated.parquet"

        # Load checkpoint if exists
        self.checkpoint = self.load_checkpoint()

        # Setup signal handler for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.interrupted = False

        # Cache for debate texts (year -> {file_path: debate_data})
        self.text_cache = {}

    def signal_handler(self, signum, frame):
        """Handle interruption gracefully"""
        print("\n\nInterrupted! Saving checkpoint...")
        self.save_checkpoint()
        print("Checkpoint saved. You can resume by running the script again.")
        self.interrupted = True
        sys.exit(0)

    def load_checkpoint(self):
        """Load checkpoint if exists"""
        if self.checkpoint_file.exists():
            print(f"Loading checkpoint from {self.checkpoint_file}...")
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"Resuming from year {checkpoint['last_completed_year'] + 1}")
            return checkpoint
        else:
            return {
                'last_completed_year': 0,
                'all_debates': [],
                'all_female_mps': set(),
                'all_male_mps': set(),
                'all_parties': set(),
                'all_constituencies': set(),
                'stats': {
                    'total_debates_processed': 0,
                    'debates_with_confirmed_mps': 0,
                    'debates_with_female': 0,
                    'debates_with_text': 0,
                    'total_text_chars': 0,
                    'by_decade': {}
                },
                'processed_years': set()
            }

    def save_checkpoint(self):
        """Save current progress"""
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(self.checkpoint, f)

    def load_year_texts(self, year):
        """Load all debate texts for a year from JSONL file"""
        if year in self.text_cache:
            return self.text_cache[year]

        year_content_file = self.content_base / str(year) / f"debates_{year}.jsonl"
        texts_by_path = {}

        if year_content_file.exists():
            try:
                with open(year_content_file, 'r') as f:
                    for line in f:
                        debate_data = json.loads(line)
                        file_path = debate_data.get('file_path', '')
                        texts_by_path[file_path] = {
                            'full_text': debate_data.get('full_text', ''),
                            'lines': debate_data.get('lines', []),
                            'content_hash': debate_data.get('content_hash', ''),
                            'extraction_timestamp': debate_data.get('extraction_timestamp', '')
                        }
                print(f"  Loaded {len(texts_by_path)} debate texts for year {year}")
            except Exception as e:
                print(f"  Warning: Could not load texts for {year}: {e}")

        self.text_cache[year] = texts_by_path
        return texts_by_path

    def extract_speeches_from_html_file(self, file_path: str) -> list:
        """
        Extract speeches from raw HTML file using structural boundaries.

        This replaces the buggy regex-based extraction that caused speaker misattribution.

        Args:
            file_path: Path to the debate from metadata

        Returns:
            List of speech dictionaries with 'speaker', 'text', 'contribution_id'
        """
        if not file_path:
            return []

        raw_html_path = get_raw_html_path(file_path, self.hansard_base)
        if raw_html_path.exists():
            return extract_speeches_from_raw_html(raw_html_path)
        return []

    def process_year(self, year, debates_df):
        """Process a single year of data with enhanced fields"""
        year_debates = []
        stats = self.checkpoint['stats']

        # Load all texts for this year
        year_texts = self.load_year_texts(year)

        stats['total_debates_processed'] += len(debates_df)

        for idx, debate in tqdm(debates_df.iterrows(), total=len(debates_df),
                                desc=f"Processing {year}", leave=False):
            if self.interrupted:
                return year_debates

            speakers = debate.get('speakers', [])
            date = debate.get('reference_date', f'{year}-01-01')
            chamber = debate.get('chamber', 'Commons')

            # Convert to list if numpy array
            if isinstance(speakers, np.ndarray):
                speakers = speakers.tolist()

            if not isinstance(speakers, list) or len(speakers) == 0:
                continue

            # Match speakers with enhanced metadata
            matched_mps = []
            female_mps = []
            male_mps = []
            speaker_details = []
            ambiguous = 0
            unmatched = 0
            unmatched_names = []

            for speaker in speakers:
                if pd.notna(speaker):
                    result = self.matcher.match_comprehensive(str(speaker), date, chamber)

                    if result['match_type'] in ['temporal_unique', 'title', 'constituency', 'titled_extraction', 'temporal_chamber_same_person', 'ambiguous_consistent_gender']:
                        if result.get('confidence', 0) >= 0.6:  # Lower threshold to include ambiguous_consistent_gender (0.6)
                            mp_name = result.get('final_match')
                            matched_mps.append(mp_name)

                            # Collect detailed speaker info
                            speaker_info = {
                                'original_name': str(speaker),
                                'matched_name': mp_name,
                                'person_id': result.get('person_id'),
                                'gender': result.get('gender'),
                                'party': result.get('party'),
                                'constituency': result.get('constituency'),
                                'confidence': result.get('confidence'),
                                'match_type': result['match_type']
                            }
                            speaker_details.append(speaker_info)

                            # Track parties and constituencies
                            if result.get('party'):
                                self.checkpoint['all_parties'].add(result.get('party'))
                            if result.get('constituency'):
                                self.checkpoint['all_constituencies'].add(result.get('constituency'))

                            if result.get('gender') == 'F':
                                female_mps.append(mp_name)
                                self.checkpoint['all_female_mps'].add(mp_name)
                            elif result.get('gender') == 'M':
                                male_mps.append(mp_name)
                                self.checkpoint['all_male_mps'].add(mp_name)
                    elif result['match_type'] == 'ambiguous':
                        ambiguous += 1
                    else:
                        unmatched += 1
                        unmatched_names.append(str(speaker))

            # Only keep debates with at least one confirmed MP
            if matched_mps:
                # Get debate text from cached data
                file_path = debate.get('file_path', '')
                debate_text_data = year_texts.get(file_path, {})
                full_text = debate_text_data.get('full_text', '')

                # Extract speech segments from raw HTML (accurate, structure-based)
                speech_segments = self.extract_speeches_from_html_file(file_path)
                if full_text:
                    stats['debates_with_text'] += 1
                    stats['total_text_chars'] += len(full_text)

                debate_id = hashlib.md5(
                    f"{file_path}_{date}".encode()
                ).hexdigest()[:16]

                decade = (year // 10) * 10

                # Enhanced debate record
                debate_record = {
                    # Basic metadata
                    'debate_id': debate_id,
                    'year': year,
                    'decade': decade,
                    'month': debate.get('month'),
                    'reference_date': date,
                    'chamber': chamber,

                    # Debate content metadata
                    'title': debate.get('title'),
                    'topic': debate.get('debate_topic'),
                    'hansard_reference': debate.get('hansard_reference'),
                    'reference_volume': debate.get('reference_volume'),
                    'reference_columns': debate.get('reference_columns'),

                    # Full text content (from extracted JSONL)
                    'debate_text': full_text,  # Full debate text, no limits
                    'text_length': len(full_text) if full_text else 0,
                    'has_text': bool(full_text),
                    'content_hash': debate_text_data.get('content_hash'),
                    'extraction_timestamp': debate_text_data.get('extraction_timestamp'),

                    # Speech segments
                    'speech_segments': speech_segments,
                    'speech_count': len(speech_segments),

                    # Speaker statistics
                    'total_speakers': len(speakers),
                    'confirmed_mps': len(matched_mps),
                    'female_mps': len(female_mps),
                    'male_mps': len(male_mps),
                    'has_female': len(female_mps) > 0,
                    'has_male': len(male_mps) > 0,

                    # Speaker details (ENHANCED)
                    'female_names': female_mps,
                    'male_names': male_mps,
                    'speaker_details': speaker_details,  # Full MP info with party, constituency
                    'ambiguous_speakers': ambiguous,
                    'unmatched_speakers': unmatched,
                    'unmatched_names': unmatched_names,  # Keep all unmatched names

                    # Gender ratio
                    'gender_ratio': len(female_mps) / (len(female_mps) + len(male_mps)) if (female_mps or male_mps) else None,

                    # Content statistics
                    'word_count': debate.get('word_count', 0),
                    'line_count': debate.get('line_count', 0),
                    'char_count': debate.get('char_count', 0),

                    # File metadata
                    'file_path': file_path,
                    'file_name': debate.get('file_name'),
                    'file_size': debate.get('file_size', 0),
                    'file_modified': debate.get('file_modified'),

                    # Processing metadata
                    'processing_timestamp': datetime.now().isoformat(),
                    'matcher_version': 'corrected_v1'
                }

                year_debates.append(debate_record)
                stats['debates_with_confirmed_mps'] += 1

                if female_mps:
                    stats['debates_with_female'] += 1

                # Update decade stats
                if decade not in stats['by_decade']:
                    stats['by_decade'][decade] = {
                        'debates_with_mps': 0,
                        'debates_with_female': 0,
                        'debates_with_text': 0,
                        'total_text_chars': 0
                    }
                stats['by_decade'][decade]['debates_with_mps'] += 1
                if female_mps:
                    stats['by_decade'][decade]['debates_with_female'] += 1
                if full_text:
                    stats['by_decade'][decade]['debates_with_text'] += 1
                    stats['by_decade'][decade]['total_text_chars'] += len(full_text)

        # Clear text cache for this year to save memory
        if year in self.text_cache:
            del self.text_cache[year]

        return year_debates

    def process_all_years(self, year_range=None, sample_mode=False):
        """Process all available years with resume capability"""

        print("=" * 70)
        print("ENHANCED GENDER ANALYSIS DATASET CREATION")
        print(f"Input: {self.content_base.parent}")
        print(f"Output: {self.output_dir}")
        print("=" * 70)

        # Get all available years from metadata
        all_years = sorted([
            int(f.stem.split('_')[1])
            for f in self.metadata_base.glob("debates_*.parquet")
            if 'master' not in f.stem
        ])

        # Filter by year range if specified
        if year_range:
            start_year, end_year = year_range
            all_years = [y for y in all_years if start_year <= y <= end_year]

        # Sample mode for testing
        if sample_mode:
            all_years = all_years[:3]  # Just process first 3 years for testing

        # Skip already processed years
        years_to_process = [y for y in all_years if y not in self.checkpoint['processed_years']]

        if not years_to_process:
            print("All years already processed!")
            return

        print(f"\nTotal years available: {len(all_years)}")
        print(f"Already processed: {len(self.checkpoint['processed_years'])}")
        print(f"Remaining to process: {len(years_to_process)}")
        print(f"Workers: {self.workers}")
        if years_to_process:
            print(f"Years range: {min(years_to_process)} - {max(years_to_process)}")
        print("=" * 70)
        print()

        # Process years in parallel
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            # Prepare arguments for each year
            args_list = [
                (year, self.content_base, self.metadata_base, self.mnis_data_path, self.hansard_base)
                for year in years_to_process
            ]

            # Process with progress bar
            with tqdm(total=len(years_to_process), desc="Processing years") as pbar:
                for result in executor.map(process_year_parallel, args_list):
                    year = result['year']

                    if 'error' in result:
                        print(f"\n✗ {year}: {result['error']}")
                        pbar.update(1)
                        continue

                    # Save year data
                    if result['debates_data']:
                        year_df = pd.DataFrame(result['debates_data'])
                        year_df.to_parquet(self.output_dir / f"debates_{year}_enhanced.parquet")

                        # Track stats (don't accumulate all data in memory)
                        for debate in result['debates_data']:
                            if debate.get('has_female'):
                                self.checkpoint['stats']['debates_with_female'] += 1
                            if debate.get('debate_text'):
                                self.checkpoint['stats']['debates_with_text'] += 1
                                self.checkpoint['stats']['total_text_chars'] += len(debate['debate_text'])

                    # Update checkpoint and stats
                    self.checkpoint['processed_years'].add(year)
                    self.checkpoint['stats']['debates_with_confirmed_mps'] += result['debates']
                    self.checkpoint['stats']['total_debates_processed'] += result.get('total_debates_in_year', result['debates'])

                    # Save checkpoint every 10 years
                    if len(self.checkpoint['processed_years']) % 10 == 0:
                        self.save_checkpoint()

                    pbar.update(1)

        # Final save (regenerate combined file from year files)
        self.save_final_dataset()

    def save_final_dataset(self):
        """Save the combined dataset and metadata (load from year files)"""
        print("\n" + "=" * 70)
        print("SAVING ENHANCED FINAL DATASET")
        print("=" * 70)

        # Load all year files and combine (saves memory vs accumulating in checkpoint)
        year_files = sorted(self.output_dir.glob("debates_*_enhanced.parquet"))

        if year_files:
            print(f"\nCombining {len(year_files)} year files...")

            # Load and combine all year files
            all_dfs = [pd.read_parquet(f) for f in year_files]
            combined_df = pd.concat(all_dfs, ignore_index=True)

            print(f"Total debates: {len(combined_df):,}")

            # Full version with text
            print("Saving full dataset with text...")
            combined_df.to_parquet(self.output_dir / "ALL_debates_enhanced_with_text.parquet")

            # Lighter version without text (for quick analysis)
            print("Saving metadata-only version...")
            combined_light = combined_df.drop(columns=['debate_text', 'speech_segments'], errors='ignore')
            combined_light.to_parquet(self.output_dir / "ALL_debates_enhanced_metadata.parquet")

            # Save metadata
            metadata = {
                'creation_date': datetime.now().isoformat(),
                'years_processed': len(self.checkpoint['processed_years']),
                'year_list': sorted(list(self.checkpoint['processed_years'])),
                'statistics': self.checkpoint['stats'],
                'total_female_mps_identified': len(self.checkpoint['all_female_mps']),
                'total_male_mps_identified': len(self.checkpoint['all_male_mps']),
                'total_parties': len(self.checkpoint['all_parties']),
                'total_constituencies': len(self.checkpoint['all_constituencies']),
                'female_mps_list': sorted(list(self.checkpoint['all_female_mps']))[:100],  # Sample
                'male_mps_list': sorted(list(self.checkpoint['all_male_mps']))[:100],  # Sample
                'parties_list': sorted(list(self.checkpoint['all_parties'])),
                'enhanced_fields': [
                    'debate_text', 'speech_segments', 'speaker_details',
                    'party', 'constituency', 'gender_ratio',
                    'unmatched_names', 'content_hash', 'text_length'
                ],
                'data_source': 'processed_fixed/content JSONL files'
            }

            with open(self.output_dir / "dataset_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            # Print summary
            stats = self.checkpoint['stats']
            total_processed = stats.get('total_debates_processed', 0)
            total_with_mps = stats.get('debates_with_confirmed_mps', 0)

            print(f"\n=== PROCESSING SUMMARY ===")
            print(f"Total debates processed: {total_processed:,}")
            print(f"Debates with confirmed MPs: {total_with_mps:,}")

            if total_with_mps > 0 and total_processed > 0:
                print(f"  → {100*total_with_mps/total_processed:.1f}% of all debates")
                print(f"\nDebates with female MPs: {stats['debates_with_female']:,}")
                print(f"  → {100*stats['debates_with_female']/total_with_mps:.1f}% of debates with MPs")
                print(f"\nDebates with extracted text: {stats['debates_with_text']:,}")
                print(f"  → {100*stats['debates_with_text']/total_with_mps:.1f}% of debates with MPs")
                print(f"Total text extracted: {stats['total_text_chars']/1e9:.2f} GB of text")

            print(f"\n=== UNIQUE ENTITIES ===")
            print(f"Female MPs identified: {len(self.checkpoint['all_female_mps'])}")
            print(f"Male MPs identified: {len(self.checkpoint['all_male_mps'])}")
            print(f"Unique parties: {len(self.checkpoint['all_parties'])}")
            print(f"Unique constituencies: {len(self.checkpoint['all_constituencies'])}")

            # Decade summary
            print("\n=== FEMALE PARTICIPATION BY DECADE ===")
            for decade in sorted(self.checkpoint['stats']['by_decade'].keys()):
                decade_stats = self.checkpoint['stats']['by_decade'][decade]
                total = decade_stats['debates_with_mps']
                female = decade_stats['debates_with_female']
                with_text = decade_stats.get('debates_with_text', 0)
                text_gb = decade_stats.get('total_text_chars', 0) / 1e9
                if total > 0:
                    print(f"{decade}s: {female:,}/{total:,} debates ({100*female/total:.1f}%) | {with_text:,} with text ({text_gb:.2f} GB)")

            print(f"\n✅ Data saved to: {self.output_dir}/")
            print("\nMain files:")
            print("  - ALL_debates_enhanced_with_text.parquet (full dataset with text)")
            print("  - ALL_debates_enhanced_metadata.parquet (lightweight, no text)")
            print("  - debates_YYYY_enhanced.parquet (individual year files)")
            print("  - dataset_metadata.json (statistics and metadata)")

            # Remove checkpoint after successful completion
            if self.checkpoint_file.exists():
                os.remove(self.checkpoint_file)
                print("\n✅ Checkpoint removed - processing complete!")

def main():
    parser = argparse.ArgumentParser(description='Create enhanced gender analysis dataset')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for dataset (default: versioned gender_analysis_v2)')
    parser.add_argument('--input-dir', default=None,
                       help='Input processed data directory (default: versioned processed_v2)')
    parser.add_argument('--year-range', nargs=2, type=int, metavar=('START', 'END'),
                       help='Process only years in range (e.g., 1900 2000)')
    parser.add_argument('--sample', action='store_true',
                       help='Sample mode - process only first 3 years for testing')
    parser.add_argument('--reset', action='store_true',
                       help='Reset checkpoint and start fresh')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: 75%% of CPU cores)')

    args = parser.parse_args()

    # Determine worker count
    if args.workers is None:
        args.workers = max(1, int(mp.cpu_count() * 0.75))

    # Handle reset
    if args.reset:
        output_path = Path(args.output_dir) if args.output_dir else Path("data/gender_analysis_enhanced")
        checkpoint_path = output_path / "checkpoint.pkl"
        if checkpoint_path.exists():
            os.remove(checkpoint_path)
            print("Checkpoint removed. Starting fresh...")

    # Create processor and run
    processor = EnhancedGenderDatasetCreator(
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        workers=args.workers
    )
    processor.process_all_years(year_range=args.year_range, sample_mode=args.sample)

if __name__ == "__main__":
    # Set multiprocessing start method for macOS compatibility
    mp.set_start_method('spawn', force=True)
    main()