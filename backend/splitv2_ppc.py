import re
from transformers import AutoTokenizer, AutoModel
from backend.utils import *

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Define regex for class and function detection for various languages
LANGUAGE_PATTERNS = {
    'python': {
        'class': r'(?:^|\n)\s*class\s+(\w+)\s*(\(.*?\))?\s*:',
        'function': r'(?m)^(?:\s*@\w+\s*)*def\s+(\w+)\s*\(.*?\)\s*:\s*([\s\S]*?)(?=^def\s|\Z)',
    },
    'javascript': {
        'class': r'(?:^|\n)\s*class\s+(\w+)\s*{',
        'function': r'(?:^|\n)\s*(?:function\s+|\b(const|let|var)\s+\w+\s*=\s*function\s*)?(\w+)\s*\(',
    },
    'java': {
        'class': r'(?:public|private|protected|abstract|final)?\s*class\s+(\w+)\s*(?:extends\s+\w+)?\s*(?:implements\s+[\w<>,\s]+)?\s*{',
        'function': r'(?:public|private|protected|static|final|abstract\s+)?(?:[\w<>\[\]]+\s+)+(\w+)\s*\([^)]*\)\s*{',
    },
    'php': {
        'class': r'(?:abstract|final)?\s*class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w, ]+)?\s*{',
        'function': r'(?:^|\n)\s*function\s+(\w+)\s*\(',
    },
    'go': {
        'class': r'type\s+(\w+)\s+struct\s*{',
        'function': r'func\s+\(.*?\)?\s*(\w+)\s*\(.*?\)\s*{',
    },
    'ruby': {
        'module': r'module\s+(\w+)\s*',
        'class': r'class\s+(\w+)\s*',
        'function': r'def\s+(\w+)\s*',
    },
}

def get_token_count(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

# Split long sections into parts based on token range, with line metadata
def split_into_parts(code, base_name, start_line):
    tokens = tokenizer.encode(code, add_special_tokens=False)
    parts = {}
    start = 0
    line_count = start_line

    i = 1
    while start < len(tokens):
        end = min(start + MAX_TOKEN_NUM, len(tokens))
        part_tokens = tokens[start:end]
        
        # Decode tokens to text and get the lines for this part
        part_text = tokenizer.decode(part_tokens)
        part_lines = part_text.strip().splitlines()
        
        first_line = f"L{line_count}"
        last_line = f"L{line_count + len(part_lines) - 1}"
        
        print(f"{base_name}_part_{i} - Token count: {len(part_tokens)} - First line: {first_line}, Last line: {last_line}")
        
        parts[f"{base_name}_part_{i}"] = {
            "content": part_text,
            "first_line": first_line,
            "last_line": last_line
        }
        
        start += MAX_TOKEN_NUM
        line_count += len(part_lines)
        i += 1

    return parts

def is_sample_too_short(code):
    token_count = get_token_count(code)
    return token_count < MIN_TOKEN_NUM

def split_code_by_patterns(code, language):
    patterns = LANGUAGE_PATTERNS.get(language, {})
    module_pattern = patterns.get('module')
    class_pattern = patterns.get('class')
    function_pattern = patterns.get('function')
    
    code_sections = {}
    global_code = ""

    # Handle Ruby modules
    if language == 'ruby' and module_pattern and re.search(module_pattern, code):
        module_matches = list(re.finditer(module_pattern, code))
        if module_matches:
            global_code = code[:module_matches[0].start()].strip()
            for i, match in enumerate(module_matches):
                module_name = match.group(1)
                module_start = match.start()
                module_body = code[module_start:module_matches[i + 1].start()].strip() if i < len(module_matches) - 1 else code[module_start:].strip()
                
                first_line = f"L{code[:module_start].count(chr(10)) + 1}"
                last_line = f"L{code[:module_start].count(chr(10)) + module_body.count(chr(10)) + 1}"
                token_count = get_token_count(module_body)
                
                if token_count < MIN_TOKEN_NUM:
                    return None
                elif token_count > MAX_TOKEN_NUM:
                    parts = split_into_parts(module_body, f"module_{module_name}", int(first_line[1:]))
                    code_sections.update(parts)
                else:
                    code_sections[f'module_{module_name}'] = {
                        "content": module_body,
                        "first_line": first_line,
                        "last_line": last_line
                    }

    # Handle classes
    elif class_pattern and re.search(class_pattern, code):
        class_matches = list(re.finditer(class_pattern, code))
        if class_matches:
            global_code = code[:class_matches[0].start()].strip()
            for i, match in enumerate(class_matches):
                class_name = match.group(1)
                class_start = match.start()
                class_body = code[class_start:class_matches[i + 1].start()].strip() if i < len(class_matches) - 1 else code[class_start:].strip()
                
                first_line = f"L{code[:class_start].count(chr(10)) + 1}"
                last_line = f"L{code[:class_start].count(chr(10)) + class_body.count(chr(10)) + 1}"
                token_count = get_token_count(class_body)
                
                if token_count < MIN_TOKEN_NUM:
                    return None
                elif token_count > MAX_TOKEN_NUM:
                    parts = split_into_parts(class_body, f"class_{class_name}", int(first_line[1:]))
                    code_sections.update(parts)
                else:
                    code_sections[f'class_{class_name}'] = {
                        "content": class_body,
                        "first_line": first_line,
                        "last_line": last_line
                    }

    # Handle functions
    elif function_pattern and re.search(function_pattern, code):
        function_matches = list(re.finditer(function_pattern, code))
        if function_matches:
            global_code = code[:function_matches[0].start()].strip()
            for i, match in enumerate(function_matches):
                function_name = match.group(1)
                function_start = match.start()
                function_body = code[function_start:function_matches[i + 1].start()].strip() if i < len(function_matches) - 1 else code[function_start:].strip()
                
                first_line = f"L{code[:function_start].count(chr(10)) + 1}"
                last_line = f"L{code[:function_start].count(chr(10)) + function_body.count(chr(10)) + 1}"
                token_count = get_token_count(function_body)
                
                if token_count < MIN_TOKEN_NUM:
                    return None
                elif token_count > MAX_TOKEN_NUM:
                    parts = split_into_parts(function_body, f"func_{function_name}", int(first_line[1:]))
                    code_sections.update(parts)
                else:
                    code_sections[f'func_{function_name}'] = {
                        "content": function_body,
                        "first_line": first_line,
                        "last_line": last_line
                    }

    # Split full code if no structures were found
    if not code_sections:
        token_count = get_token_count(code)
        first_line = "L1"
        last_line = f"L{code.count(chr(10)) + 1}"
        
        if token_count > MAX_TOKEN_NUM:
            return split_into_parts(code, "code", 1)
        elif token_count >= MIN_TOKEN_NUM:
            return {"full_code": {
                "content": code,
                "first_line": None,
                "last_line": None
            }}
        else:
            return None
    
    return code_sections

def preprocess_code(code, language):
    if is_sample_too_short(code):
        return None
    
    code_sections = split_code_by_patterns(code, language)
    return code_sections
