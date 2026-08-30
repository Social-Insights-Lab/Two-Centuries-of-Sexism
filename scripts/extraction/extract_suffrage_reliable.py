#!/usr/bin/env python3
"""
Extract reliable suffrage dataset based on validation findings.

Based on large-sample validation (n=300):
- Tier 1: 95% precision (explicit suffrage terms)
- Tier 2 MEDIUM: 25.7% precision (women + voting in proximity)
- Tier 2 LOW/OFF: Not reliable

This extracts only HIGH and MEDIUM confidence speeches.
"""

import pandas as pd
from pathlib import Path
import re


class ReliableSuffrageExtractor:
    """Extract only reliable suffrage speeches."""

    def __init__(self, year_range=(1900, 1935), data_dir='data-hansard/derived_complete'):
        self.year_range = year_range
        self.data_dir = Path(data_dir)

    def extract_reliable(self, chamber='Commons', min_word_count=50):
        """
        Extract only reliable suffrage speeches:
        - TIER 1 (HIGH): Explicit suffrage terms (95% precision)
        - TIER 2 (MEDIUM): Women + voting in close proximity (validated)

        Args:
            chamber: Filter to specific chamber
            min_word_count: Minimum words to count as substantive speech

        Returns:
            pd.DataFrame with 'confidence_level' column (HIGH or MEDIUM)
        """
        print(f"\nReliable suffrage extraction: {self.year_range[0]}-{self.year_range[1]} ({chamber})")
        print(f"Minimum speech length: {min_word_count} words\n")

        all_speeches = []

        for year in range(self.year_range[0], self.year_range[1] + 1):
            speech_file = self.data_dir / 'speeches_complete' / f'speeches_{year}.parquet'

            if not speech_file.exists():
                continue

            speeches_df = pd.read_parquet(speech_file)

            # Filter chamber and length
            if chamber:
                speeches_df = speeches_df[speeches_df['chamber'] == chamber]
            speeches_df = speeches_df[speeches_df['word_count'] >= min_word_count]

            # HIGH: Explicit suffrage terms (Tier 1)
            high_pattern = (
                'women.*suffrage|female suffrage|suffrage.*women|'
                'votes for women|suffragette|suffragist|'
                'enfranchise.*women|women.*enfranchise|'
                'equal franchise|'
                'representation of the people.*women|'
                'sex disqualification|'
                'women.*social.*political.*union'
            )

            high_match = speeches_df['text'].str.contains(high_pattern, case=False, na=False)

            # MEDIUM: Women + voting in close proximity
            # This is more expensive, so only check speeches not already HIGH
            medium_candidates = speeches_df[~high_match].copy()
            medium_match_indices = []

            if len(medium_candidates) > 0:
                print(f"  {year}: Checking {len(medium_candidates)} speeches for MEDIUM matches...", end='', flush=True)

                for idx, row in medium_candidates.iterrows():
                    if self._is_medium_confidence(row['text']):
                        medium_match_indices.append(idx)

                print(f" found {len(medium_match_indices)}")

            # Combine HIGH and MEDIUM
            high_count = high_match.sum()
            medium_count = len(medium_match_indices)

            if high_count > 0 or medium_count > 0:
                # Get HIGH speeches
                year_speeches = []
                if high_count > 0:
                    high_speeches = speeches_df[high_match].copy()
                    high_speeches['confidence_level'] = 'HIGH'
                    year_speeches.append(high_speeches)

                # Get MEDIUM speeches
                if medium_count > 0:
                    medium_speeches = speeches_df.loc[medium_match_indices].copy()
                    medium_speeches['confidence_level'] = 'MEDIUM'
                    year_speeches.append(medium_speeches)

                year_combined = pd.concat(year_speeches, ignore_index=True)

                print(f"  {year}: {len(year_combined):4} speeches "
                      f"(HIGH: {high_count:3}, MEDIUM: {medium_count:3})")

                all_speeches.append(year_combined)
            else:
                print(f"  {year}: No speeches found")

        if all_speeches:
            speeches_combined = pd.concat(all_speeches, ignore_index=True)

            print(f"\n{'='*70}")
            print(f"EXTRACTION COMPLETE")
            print(f"{'='*70}")
            print(f"Total reliable speeches: {len(speeches_combined):,}")
            print(f"  HIGH (explicit suffrage, ~95% precision): {(speeches_combined['confidence_level'] == 'HIGH').sum():,}")
            print(f"  MEDIUM (women's voting, ~26% precision): {(speeches_combined['confidence_level'] == 'MEDIUM').sum():,}")
            print(f"Date range: {speeches_combined['date'].min()} to {speeches_combined['date'].max()}")

            # Estimated true positives
            high_count = (speeches_combined['confidence_level'] == 'HIGH').sum()
            medium_count = (speeches_combined['confidence_level'] == 'MEDIUM').sum()
            estimated_true = high_count * 0.95 + medium_count * 0.257

            print(f"\nEstimated true positives: ~{estimated_true:.0f} ({estimated_true/len(speeches_combined)*100:.1f}%)")

            return speeches_combined
        else:
            print("\nNo speeches found!")
            return None

    def _is_medium_confidence(self, text):
        """
        Check if speech is MEDIUM confidence: women/female + voting terms in close proximity.

        Based on validation: searches for women/female within 25 words of voting-related terms.
        """
        text_lower = text.lower()

        # Define voting-related patterns
        voting_patterns = [
            r'vote', r'voting', r'voter', r'voters',
            r'electoral', r'electorate',
            r'franchise', r'enfranchise',
            r'representation',
        ]

        # Split into words
        words = text_lower.split()

        # Look for women/female
        for i, word in enumerate(words):
            if 'women' in word or 'female' in word:
                # Check surrounding context (25 words before and after)
                start = max(0, i - 25)
                end = min(len(words), i + 25)
                context = ' '.join(words[start:end])

                # Check if any voting term appears in context
                for pattern in voting_patterns:
                    if re.search(pattern, context):
                        return True

        return False


def run_extraction():
    """Run the reliable extraction."""
    print("="*70)
    print("RELIABLE SUFFRAGE EXTRACTION")
    print("Based on n=300 validation: HIGH (~95%) + MEDIUM (~26%) only")
    print("="*70)

    extractor = ReliableSuffrageExtractor(year_range=(1803, 2005))
    speeches = extractor.extract_reliable(chamber='Commons', min_word_count=50)

    if speeches is not None:
        # Three-era analysis
        print(f"\n{'='*70}")
        print("THREE-ERA ANALYSIS")
        print(f"{'='*70}")

        pre_1918 = speeches[speeches['year'] < 1918]
        partial = speeches[(speeches['year'] >= 1918) & (speeches['year'] < 1928)]
        post_1928 = speeches[speeches['year'] >= 1928]

        for era_name, era_data in [
            ('Pre-1918 (no women vote)', pre_1918),
            ('1918-1927 (partial suffrage)', partial),
            ('Post-1928 (equal suffrage)', post_1928)
        ]:
            matched_era = era_data[era_data['matched_mp'] == True]
            high = era_data[era_data['confidence_level'] == 'HIGH']
            medium = era_data[era_data['confidence_level'] == 'MEDIUM']

            print(f"\n{era_name}:")
            print(f"  Total speeches: {len(era_data):,}")
            print(f"    HIGH: {len(high):,}, MEDIUM: {len(medium):,}")
            print(f"  Matched to MPs: {len(matched_era):,} ({len(matched_era)/len(era_data)*100:.1f}%)")

            if len(matched_era) > 0:
                gender = matched_era['gender'].value_counts()
                print(f"  Male: {gender.get('M', 0):,}, Female: {gender.get('F', 0):,}")

        # Peak years analysis
        print(f"\n{'='*70}")
        print("PEAK YEARS")
        print(f"{'='*70}")

        by_year = speeches.groupby(['year', 'confidence_level']).size().unstack(fill_value=0)
        by_year['total'] = by_year.sum(axis=1)

        # Show years with 50+ speeches
        peak_years = by_year[by_year['total'] >= 50].sort_values('total', ascending=False)
        print(f"\nTop years (50+ speeches):")
        print(peak_years.head(15).to_string())

        # Save results
        print(f"\n{'='*70}")
        print("SAVING RESULTS")
        print(f"{'='*70}")

        output_dir = Path('outputs/suffrage_reliable')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main dataset
        speeches.to_parquet(output_dir / 'speeches_reliable.parquet')

        # Save confidence-level specific versions
        high = speeches[speeches['confidence_level'] == 'HIGH']
        medium = speeches[speeches['confidence_level'] == 'MEDIUM']

        high.to_parquet(output_dir / 'speeches_high_confidence.parquet')
        medium.to_parquet(output_dir / 'speeches_medium_confidence.parquet')

        # CSV samples for inspection
        speeches.head(500).to_csv(output_dir / 'speeches_sample.csv', index=False)
        high.head(100).to_csv(output_dir / 'high_confidence_sample.csv', index=False)
        medium.head(100).to_csv(output_dir / 'medium_confidence_sample.csv', index=False)

        # Save summary statistics
        summary = {
            'total_speeches': len(speeches),
            'high_confidence': len(high),
            'medium_confidence': len(medium),
            'estimated_true_positives': int(len(high) * 0.95 + len(medium) * 0.257),
            'date_range': f"{speeches['date'].min()} to {speeches['date'].max()}",
            'year_range': f"{speeches['year'].min()} to {speeches['year'].max()}",
            'matched_to_mps': int((speeches['matched_mp'] == True).sum()),
            'match_rate': f"{(speeches['matched_mp'] == True).mean() * 100:.1f}%",
        }

        with open(output_dir / 'SUMMARY.txt', 'w') as f:
            f.write("RELIABLE SUFFRAGE DATASET SUMMARY\n")
            f.write("="*70 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

        print(f"\nSaved to {output_dir}/")
        print(f"  - speeches_reliable.parquet ({len(speeches):,} speeches)")
        print(f"  - speeches_high_confidence.parquet ({len(high):,} speeches)")
        print(f"  - speeches_medium_confidence.parquet ({len(medium):,} speeches)")
        print(f"  - CSV samples for inspection")
        print(f"  - SUMMARY.txt with key statistics")

        return speeches

    return None


if __name__ == '__main__':
    speeches = run_extraction()
