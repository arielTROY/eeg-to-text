# Paper Improvement Log

## Round 0 (baseline)
**File**: main_round0_original.pdf (150 KB)
**Issues identified**:
- Table 1: 22ch no-embed reported as 68.5% (wrong — should be 72.5%)
- Abstract ±6.3% — wrong SD/SE label
- "+4.1%" gain at 22ch was subject-1 only; 9-subject mean is +2.6%
- Missing `\IEEEpeerreviewmaketitle`
- `\multicolumn` footnote inside tabular body — LaTeX error

## Round 1
**File**: main_round1.pdf (151 KB)
**Fixes**:
- Corrected Table 1 22ch no-embed: 72.5% ± 14.6%
- Fixed ±4.4% SE in abstract
- Disambiguated "+2.6% mean / +4.1% subject 1" for 22ch result
- Carrara: added ✝ footnote explaining 6/9 subjects
- Added `d=4` neurophysiological justification in Methods

## Round 2
**File**: main_round2.pdf (152 KB)
**Fixes**:
- Fixed multicolumn footnote → minipage
- Rewrote Conclusion to tell a forward-looking story (not just repeat abstract)
- Added open-source code note
- Added `\IEEEpeerreviewmaketitle`

## Final score: 8/10
**Remaining before submission**:
1. Add Fig 1 (pipeline schematic) — manual creation needed
2. Verify page count (likely 7-8pp, within TNSRE 10pp limit)
3. Check all citation DOIs
4. Author affiliation and funding acknowledgment
