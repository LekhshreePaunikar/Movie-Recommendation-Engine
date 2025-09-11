// static/app.js
// Populates <datalist> only after the user starts typing and never auto-corrects input

(() => {
  // Grab the key DOM nodes we interact with.
  const input = document.getElementById("movie");
  const form = document.getElementById("search-form");
  const datalist = document.getElementById("titles");
  const SUGGEST_URL = "/api/suggest";
  let inFlight; // Tracks the current request so we can cancel it
  let lastQuery = "";

  // Debounce helper: limit how often we hit the API while typing
  const debounce = (fn, ms = 200) => {
    let t;
    return (...args) => {
      clearTimeout(t);
      t = setTimeout(() => fn(...args), ms);
    };
  };

  // Remove all <option> items from the datalist
  const clearOptions = () => {
    while (datalist.firstChild) datalist.removeChild(datalist.firstChild);
  };

  // Render a fresh set of <option> items into the datalist
  const renderOptions = (items) => {
    clearOptions();
    items.forEach((title) => {
      const opt = document.createElement("option");
      opt.value = title;
      datalist.appendChild(opt);
    });
  };

  // Call the backend for suggestions; cancel any previous in-flight request
  const fetchSuggestions = async (q) => {
    // cancel previous in-flight request
    if (inFlight) inFlight.abort();
    inFlight = new AbortController();

    const url = new URL(SUGGEST_URL, window.location.origin);
    url.searchParams.set("q", q);
    const res = await fetch(url.toString(), { signal: inFlight.signal });
    if (!res.ok) return [];

    const data = await res.json();
    return Array.isArray(data?.suggestions) ? data.suggestions : [];
  };

  // On input: skip empty/duplicate queries, then fetch and render suggestions
  const onInput = debounce(async () => {
    const q = (input.value || "").trim();
    if (q.length === 0) {
      lastQuery = "";
      clearOptions();
      return;
    }
    // avoid duplicate calls for same query
    if (q === lastQuery) return;
    lastQuery = q;

    try {
      const suggestions = await fetchSuggestions(q);
      renderOptions(suggestions);
    } catch {
      // Network/abort errors are fine: keep whatever options we already have
    }
  }, 200);

  // Let Enter submit the form using the exact value typed (no auto-correction)
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  input.addEventListener("input", onInput);
})();
