import sys, io, re

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def is_text_comment(text):
    s = str(text or "").strip()
    if not s:
        return False
    stripped = re.sub(
        r'[\U0001F600-\U0001F64F'
        r'\U0001F300-\U0001F5FF'
        r'\U0001F680-\U0001F6FF'
        r'\U0001F1E0-\U0001F1FF'
        r'\U00002702-\U000027B0'
        r'\U000024C2-\U0001F251'
        r'\U0001F900-\U0001F9FF'
        r'\U0001FA00-\U0001FA6F'
        r'\U0001FA70-\U0001FAFF'
        r'\U00002600-\U000026FF'
        r'\U0000FE00-\U0000FE0F'
        r'\U0000200D'
        r'\U00000020'
        r']+', '', s
    )
    if not any(ch.isalnum() for ch in stripped):
        return False
    without_hashtags = re.sub(r'#\w+', '', s).strip()
    without_hashtags = re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251'
        r'\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF'
        r'\U00002600-\U000026FF\U0000FE00-\U0000FE0F\U0000200D\s]+', '', without_hashtags
    )
    if not any(ch.isalnum() for ch in without_hashtags):
        return False
    return True


tests = [
    ('Hello world', True),
    ('\U0001F525\U0001F525\U0001F525', False),
    ('#freepalestine', False),
    ('#freepalestine \U0001F1F5\U0001F1F8', False),
    ('#CocaCola love it!', True),
    ('Great Day for a Coca Cola', True),
    ('\U0001F60D\U0001F60D\U0001F60D', False),
    ('E', True),
    ('Line1\nLine2\nLine3', True),
    ('#boycotworldcup', False),
    ('#wearewithpalestine\U0001F1F5\U0001F1F8\u2764', False),
    ('#Fun \u2763\uFE0F', False),
    ('#Colaboykot', False),
    ('Free Palestine \u2764\uFE0F', True),
]

all_ok = True
for text, expected in tests:
    result = is_text_comment(text)
    status = 'OK' if result == expected else 'FAIL'
    if result != expected:
        all_ok = False
    print(f'{status}: is_text_comment({repr(text[:40])}) = {result} (expected {expected})')

# Test newline removal
raw = 'Line1\nLine2\r\nLine3'
cleaned = re.sub(r'[\r\n]+', ' ', raw).strip()
has_newline = '\n' in cleaned or '\r' in cleaned
print(f'\nNewline test: {"OK" if not has_newline else "FAIL"} -> {repr(cleaned)}')

print(f'\nAll tests passed: {all_ok}')
