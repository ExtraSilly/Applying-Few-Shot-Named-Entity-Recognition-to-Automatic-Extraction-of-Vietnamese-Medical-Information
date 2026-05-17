/* ── DOM refs ─────────────────────────────────────────────────── */
const exampleSelect   = document.getElementById('example-select');
const inputText       = document.getElementById('input-text');
const predictBtn      = document.getElementById('predict-btn');
const resultSection   = document.getElementById('result-section');
const loadingDiv      = document.getElementById('loading');
const resultContent   = document.getElementById('result-content');
const highlightedBox  = document.getElementById('highlighted-text');
const entityGrid      = document.getElementById('entity-grid');
const entityCount     = document.getElementById('entity-count');
const entitySection   = document.getElementById('entity-section');
const noEntityMsg     = document.getElementById('no-entity-msg');
const errorMsg        = document.getElementById('error-msg');
const segmentedSection = document.getElementById('segmented-section');
const segmentedText   = document.getElementById('segmented-text');

/* ── Chon cau vi du ───────────────────────────────────────────── */
exampleSelect.addEventListener('change', () => {
  if (exampleSelect.value) inputText.value = exampleSelect.value;
});

/* ── Submit ───────────────────────────────────────────────────── */
predictBtn.addEventListener('click', runPredict);
inputText.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runPredict(); }
});

async function runPredict() {
  const text = inputText.value.trim();
  if (!text) { showError('Vui lòng nhập văn bản trước.'); return; }

  // Show loading
  resultSection.style.display = 'block';
  loadingDiv.style.display    = 'flex';
  resultContent.style.display = 'none';
  errorMsg.style.display      = 'none';
  predictBtn.disabled         = true;

  try {
    const res  = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text }),
    });
    const data = await res.json();

    if (!res.ok) { showError(data.error || 'Lỗi server.'); return; }

    renderResult(data);
  } catch (err) {
    showError('Không kết nối được server: ' + err.message);
  } finally {
    loadingDiv.style.display = 'none';
    predictBtn.disabled      = false;
  }
}

/* ── Render highlighted text ──────────────────────────────────── */
function renderResult(data) {
  const { word_tags, entities, colors, entity_vi, segmented, auto_segment } = data;

  // Hien thi van ban sau tach tu (neu underthesea co san)
  if (auto_segment && segmented) {
    segmentedText.textContent      = segmented;
    segmentedSection.style.display = 'block';
  } else {
    segmentedSection.style.display = 'none';
  }

  // Highlighted text
  const html = buildHighlightedHTML(word_tags, colors, entity_vi);
  highlightedBox.innerHTML = html;

  // Entity cards
  entityGrid.innerHTML = '';
  if (entities.length > 0) {
    entitySection.style.display = 'block';
    noEntityMsg.style.display   = 'none';
    entityCount.textContent     = entities.length;

    entities.forEach(ent => {
      const card = document.createElement('div');
      card.className = 'entity-card';
      card.style.background = ent.color;
      card.innerHTML = `<strong>${escHtml(ent.text)}</strong>
        <small>${escHtml(ent.type_vi)} &nbsp;·&nbsp; <code>${escHtml(ent.type)}</code></small>`;
      entityGrid.appendChild(card);
    });
  } else {
    entitySection.style.display = 'none';
    noEntityMsg.style.display   = 'block';
  }

  resultContent.style.display = 'block';
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function buildHighlightedHTML(wordTags, colors, entityVi) {
  const parts = [];
  let i = 0;
  while (i < wordTags.length) {
    const { word, tag } = wordTags[i];
    if (tag.startsWith('B-')) {
      const etype   = tag.slice(2);
      const spanArr = [word];
      let j = i + 1;
      while (j < wordTags.length && wordTags[j].tag === `I-${etype}`) {
        spanArr.push(wordTags[j].word);
        j++;
      }
      const color   = colors[etype]  || '#eee';
      const labelVi = entityVi[etype] || etype;
      const text    = spanArr.join(' ');
      parts.push(
        `<span class="entity-span" style="background:${color}" title="${escHtml(labelVi)}">`
        + `${escHtml(text)}<sup>${escHtml(labelVi)}</sup></span>`
      );
      i = j;
    } else {
      parts.push(`<span>${escHtml(word)}</span>`);
      i++;
    }
  }
  return parts.join(' ');
}

/* ── Helpers ──────────────────────────────────────────────────── */
function showError(msg) {
  resultSection.style.display = 'block';
  loadingDiv.style.display    = 'none';
  resultContent.style.display = 'none';
  errorMsg.style.display      = 'block';
  errorMsg.textContent        = msg;
}

function escHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
