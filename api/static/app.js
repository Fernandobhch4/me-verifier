const $ = sel => document.querySelector(sel);

const fileInput = $("#file");
const btnVerify = $("#btnVerify");
const preview   = $("#preview");
const previewWrap = $("#previewWrap");
const resultBox = $("#result");
const errorBox  = $("#error");
const spinner   = $("#spinner");

let currentFile = null;

fileInput.addEventListener("change", () => {
  errorBox.classList.add("hidden");
  resultBox.classList.add("hidden");
  const f = fileInput.files[0];
  if (!f) { btnVerify.disabled = true; return; }
  if (!["image/jpeg","image/png"].includes(f.type)) {
    showError("Formato no válido (usa JPG o PNG).");
    btnVerify.disabled = true;
    return;
  }
  currentFile = f;
  const url = URL.createObjectURL(f);
  preview.src = url;
  previewWrap.classList.remove("hidden");
  btnVerify.disabled = false;
});

btnVerify.addEventListener("click", async () => {
  if (!currentFile) return;
  setLoading(true);
  try {
    const fd = new FormData();
    fd.append("image", currentFile);
    const res = await fetch("/verify", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) {
      showError(data?.error || "Error al verificar");
      return;
    }
    showResult(data);
  } catch (e) {
    showError("Error de red o servidor.");
  } finally {
    setLoading(false);
  }
});

function setLoading(loading) {
  btnVerify.disabled = loading;
  spinner.classList.toggle("hidden", !loading);
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("hidden");
  resultBox.classList.add("hidden");
}

function showResult(data) {
  errorBox.classList.add("hidden");
  const { is_me, score, threshold, timing_ms } = data;
  resultBox.innerHTML = `
    <div class="badge ${is_me ? "yes" : "no"}">
      ${is_me ? "ES FERNANDO" : "NO ES FERNANDO"}
    </div>
    <div class="kv">score: <b>${Number(score).toFixed(4)}</b></div>
    <div class="kv">threshold: <b>${Number(threshold).toFixed(2)}</b></div>
    <div class="kv">latencia: <b>${timing_ms} ms</b></div>
  `;
  resultBox.classList.remove("hidden");
}
