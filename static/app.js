const $ = (s) => document.querySelector(s);
const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const money = (v) => "$" + Number(v).toLocaleString("en-US");

// tabs
document.querySelectorAll(".tab").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".tab,.panel").forEach((e) => e.classList.remove("active"));
  b.classList.add("active"); $("#" + b.dataset.tab).classList.add("active");
});

// autocomplete (debounced)
let timer;
$("#q").addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(async () => {
    const q = $("#q").value.trim(), box = $("#suggest");
    if (!q) { box.style.display = "none"; return; }
    const items = await (await fetch("/api/search?q=" + encodeURIComponent(q))).json();
    box.innerHTML = items.map((m) => `<div data-t="${esc(m.title)}">${esc(m.title)} <small>${m.year ?? ""}</small></div>`).join("");
    box.style.display = items.length ? "block" : "none";
  }, 200);
});
$("#suggest").addEventListener("click", (e) => {
  const t = e.target.closest("div[data-t]"); if (!t) return;
  $("#q").value = t.dataset.t; $("#suggest").style.display = "none"; recommend();
});
$("#q").addEventListener("keydown", (e) => { if (e.key === "Enter") { $("#suggest").style.display = "none"; recommend(); } });
$("#go").onclick = recommend;

function card(m, isQuery = false) {
  return `<div class="card ${isQuery ? "query" : ""}">
    ${m.similarity !== undefined ? `<span class="badge">${Math.round(m.similarity * 100)}% match</span>` : ""}
    <h3>${esc(m.title)}</h3>
    <div class="meta">${m.year ?? "—"} · ★ ${m.rating} (${m.votes.toLocaleString()} votes)${m.director ? " · " + esc(m.director) : ""}</div>
    <div>${m.genres.map((g) => `<span class="tag">${esc(g)}</span>`).join("")}</div>
    <p>${esc(m.overview)}</p></div>`;
}

async function recommend() {
  const title = $("#q").value.trim();
  $("#msg").textContent = ""; $("#results").innerHTML = ""; $("#query").innerHTML = "";
  if (!title) { $("#msg").textContent = "Enter a movie title first."; return; }
  const params = new URLSearchParams({ title, n: $("#n").value, genre: $("#genre").value });
  const res = await fetch("/api/recommend?" + params);
  const data = await res.json();
  if (!res.ok) {
    $("#msg").textContent = data.error + (data.suggestions?.length ? " Did you mean: " + data.suggestions.join(", ") + "?" : "");
    return;
  }
  $("#query").innerHTML = "<h4>Because you picked</h4>" + card(data.query, true) + "<h4>You might also like</h4>";
  $("#results").innerHTML = data.recommendations.map((m) => card(m)).join("") || "<p>No matches for that filter.</p>";
}

// revenue prediction
$("#revform").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  const body = {
    budget: +f.get("budget"), runtime: +f.get("runtime"), year: +f.get("year"), month: +f.get("month"),
    n_companies: +f.get("n_companies") || 2, language: f.get("language"), genres: f.getAll("genres"),
  };
  const res = await fetch("/api/predict-revenue", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const d = await res.json();
  $("#revout").innerHTML = res.ok
    ? `<div class="card"><div class="meta">Estimated worldwide revenue</div><div class="big">${money(d.predicted_revenue)}</div>
       <div class="meta">Likely range (10th–90th percentile of trees): ${money(d.low)} – ${money(d.high)}<br>
       Model R² (log scale, held-out): ${d.model_r2_log}. Treat as a rough estimate.</div></div>`
    : `<p id="msg">${esc(d.error)}</p>`;
});
