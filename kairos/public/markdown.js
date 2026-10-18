const escapeHtml = (text) => String(text ?? '').replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);

function inlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
}

const splitTableRow = (line) => line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim());
const isTableRow = (line) => line.includes('|') && splitTableRow(line).length > 1;
const isSeparatorRow = (line) => isTableRow(line) && splitTableRow(line).every((cell) => /^:?-{3,}:?$/.test(cell));

function tableToMetricList(lines, start) {
  const headers = splitTableRow(lines[start]);
  const rows = [];
  let i = start + 2;
  while (i < lines.length && isTableRow(lines[i])) {
    rows.push(splitTableRow(lines[i]));
    i += 1;
  }
  const items = rows.map((row) => `<li>${headers.map((header, j) => `<span><strong>${inlineMarkdown(header)}:</strong> ${inlineMarkdown(row[j] ?? '')}</span>`).join('')}</li>`).join('');
  return { html: `<ul class="metric-list">${items}</ul>`, next: i };
}

export function markdownToHtml(markdown = '') {
  const lines = String(markdown || '').split(/\r?\n/);
  const out = [];
  let inCode = false;
  let inList = false;
  let code = [];
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (line.startsWith('```')) {
      if (inCode) {
        out.push(`<pre><code>${escapeHtml(code.join('\n'))}</code></pre>`);
        code = [];
      }
      inCode = !inCode;
      continue;
    }
    if (inCode) { code.push(line); continue; }
    if (isTableRow(line) && isSeparatorRow(lines[i + 1] || '')) {
      if (inList) { out.push('</ul>'); inList = false; }
      const list = tableToMetricList(lines, i);
      out.push(list.html);
      i = list.next - 1;
      continue;
    }
    const item = line.match(/^[-*]\s+(.+)/);
    if (item) {
      if (!inList) { out.push('<ul>'); inList = true; }
      out.push(`<li>${inlineMarkdown(item[1])}</li>`);
      continue;
    }
    if (inList) { out.push('</ul>'); inList = false; }
    if (!line.trim()) { out.push(''); continue; }
    const heading = line.match(/^(#{1,3})\s+(.+)/);
    if (heading) { out.push(`<h${heading[1].length}>${inlineMarkdown(heading[2])}</h${heading[1].length}>`); continue; }
    out.push(`<p>${inlineMarkdown(line)}</p>`);
  }
  if (inCode) out.push(`<pre><code>${escapeHtml(code.join('\n'))}</code></pre>`);
  if (inList) out.push('</ul>');
  return out.join('\n');
}

export function promptResultMarkdown(data = {}) {
  if (data.type === 'confirmation_required') {
    return `## Confirm deployment\n\n- Tool: \`${data.tool}\`\n- Namespace: \`${data.namespace}\`\n- Template: \`${data.templateName}\`\n\n\`\`\`json\n${JSON.stringify(data.parameters || {}, null, 2)}\n\`\`\``;
  }
  const parts = [data.message || JSON.stringify(data, null, 2)];
  if (data.llmError) parts.push(`\n\n---\n\n**LLM fallback reason**\n\n\`\`\`json\n${data.llmError}\n\`\`\``);
  return parts.join('');
}
