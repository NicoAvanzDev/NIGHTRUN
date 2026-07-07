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

// Docs table-of-contents: highlight the section currently in view.
const toc = document.querySelector(".toc");
if (toc && "IntersectionObserver" in window) {
  const links = new Map(
    [...toc.querySelectorAll("a[href^='#']")].map((a) => [a.getAttribute("href").slice(1), a])
  );
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        const link = links.get(e.target.id);
        if (link && e.isIntersecting) {
          links.forEach((l) => l.classList.remove("active"));
          link.classList.add("active");
        }
      });
    },
    { rootMargin: "0px 0px -70% 0px" }
  );
  links.forEach((_, id) => {
    const el = document.getElementById(id);
    if (el) observer.observe(el);
  });
}
