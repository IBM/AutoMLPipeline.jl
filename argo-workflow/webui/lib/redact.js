const SECRET_PATTERNS = [
  /(authorization:\s*bearer\s+)[^\s"']+/gi,
  /((?:api[_-]?key|token|password|passwd|secret)\s*[:=]\s*)[^\s"'&]+/gi,
  /(https?:\/\/[^\s:]+:)[^@\s]+(@)/gi,
  /(AKIA[0-9A-Z]{16})/g
];

export function redact(value) {
  if (value == null) return value;
  const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  return SECRET_PATTERNS.reduce((acc, re) => acc.replace(re, '$1[REDACTED]$2'), text);
}

export function safeJson(value) {
  return JSON.parse(redact(value));
}
