// NightRun site enhancements. Everything here is optional: the site is
// fully usable with JavaScript disabled (navigation is CSS-only).

// Copy buttons on command blocks marked with data-copy.
document.querySelectorAll("pre[data-copy]").forEach((pre) => {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "copy-btn";
  btn.textContent = "copy";
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(pre.querySelector("code").innerText);
      btn.textContent = "copied";
    } catch {
      btn.textContent = "select + ctrl-c";
    }
    setTimeout(() => (btn.textContent = "copy"), 1600);
  });
  pre.appendChild(btn);
});

