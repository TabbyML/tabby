# Bug Report: README.md Typo

## Summary
Found and fixed a typo in the README.md file in the "What's New" section.

## Bug Details
**Location:** Line 59 (06/13/2024 entry in What's New section)

**Original Text:**
```
Come and they the latest **chat in side-panel** and **editing via chat command**!
```

**Issue:** Grammar error - "they" should be "try"

**Fixed Text:**
```
Come and try the latest **chat in side-panel** and **editing via chat command**!
```

## Impact
- Minor grammar/readability issue in documentation
- Does not affect functionality

## Files Changed
- `README.md` (1 line modified)

## Additional Note
- `package.json`: pinned `pnpm` engine to `>=8 <9` because pnpm 9.x is not supported and causes installation failures on some environments. See issue #4499 for context.
