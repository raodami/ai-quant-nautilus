STRATEGY_SCHEMA = {
    "type": "object",
    "required": ["name", "rationale", "code", "params", "expected_edge"],
    "properties": {
        "name": {
            "type": "string",
            "pattern": "^gen_[a-z0-9_]+$",
            "description": "策略名, gen_ 前缀, 小写下划线分隔",
        },
        "rationale": {
            "type": "string",
            "minLength": 20,
            "description": "经济学逻辑说明",
        },
        "code": {
            "type": "string",
            "description": "完整 Python 类代码",
        },
        "params": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
        "expected_edge": {
            "type": "string",
            "description": "预期优势与市场状态",
        },
    },
    "strict": True,
}


def validate_schema_output(result: dict) -> tuple[bool, list[str]]:
    """Validate LLM output against schema."""
    import re
    errors = []

    required = ["name", "rationale", "code", "params", "expected_edge"]
    for field in required:
        if field not in result:
            errors.append(f"缺少必填字段: {field}")

    if errors:
        return False, errors

    name = result.get("name", "")
    if not re.match(r"^gen_[a-z0-9_]+$", name):
        errors.append(f"name 格式错误: {name} (应为 gen_xxx 格式)")

    code = result.get("code", "")
    if not code or len(code.strip()) < 50:
        errors.append("code 内容过短")

    return len(errors) == 0, errors
