"""Fix double-encoded UTF-8 in app.html"""
import re

filepath = r"d:\TESTING PROTOTYPES\PROTOTYPE 5\templates\app.html"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix all known mojibake patterns
replacements = {
    '\u00e2\u0080\u0094': '-',      # em dash (—) double-encoded
    '\u00e2\u0080\u0093': '-',      # en dash (–) double-encoded
    '\u00e2\u0080\u00a6': '...',    # ellipsis (…) double-encoded
    '\u00c3\u00a2\u00c2\u0080\u00c2\u0094': '-',  # triple-encoded em dash
    '\u00c3\u00a2\u00c2\u0080\u00c2\u00a6': '...', # triple-encoded ellipsis
}

for bad, good in replacements.items():
    if bad in content:
        count = content.count(bad)
        content = content.replace(bad, good)
        print(f"Replaced {count} occurrences of mojibake -> '{good}'")

# Fix the garbled emoji for fingerspelling label
# The 📤 emoji (U+1F524) when double-encoded appears as various mojibake
# Find the pattern in the video overlay line
content = re.sub(
    r"`[^\`]*\$\{item\.word\}`\s*:\s*item\.word",
    "`[${item.word}]` : item.word",
    content,
    count=1
)

# More targeted: fix the specific line with the letter overlay
old_letter_line = None
lines = content.split('\n')
for i, line in enumerate(lines):
    if "item.type === 'letter'" in line and 'wordOverlay' in line:
        # Replace the entire expression with clean ASCII
        lines[i] = "                this.wordOverlay.textContent = item.type === 'letter' ? '[' + item.word + ']' : item.word;"
        print(f"Fixed fingerspelling label on line {i+1}")
        break

content = '\n'.join(lines)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Done! File saved.")

# Verify
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
    
# Check for remaining issues
issues = []
for pattern in ['\u00e2\u0080', '\u00c3\u00a2']:
    if pattern in content:
        issues.append(f"Still contains: {repr(pattern)}")

if issues:
    print(f"WARNING: {len(issues)} remaining issues")
    for i in issues:
        print(f"  {i}")
else:
    print("All encoding issues resolved!")
