def format_text(s, width=20):
    if ':' in s:
        return format_text(s[:s.find(':')], width=width)

    words = [x.strip() for x in s.strip().replace('hr', 'HR').replace('Hr', 'HR').replace('heart rate', 'HR').split()]
    words[0] = ''.join([w if i > 0 else w.upper() for i, w in enumerate(words[0])])

    idx = 0
    formatted = ''
    line = ''
    while idx < len(words):
        if len(line) + len(words[idx]) > width:
            formatted += line + '\n'
            line = words[idx]
        else:
            line += ' ' + words[idx]
        idx += 1
    formatted += line
    return formatted.strip()