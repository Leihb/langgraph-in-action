// 把 mdBook 渲染出来的 ```mermaid 代码块换成 Mermaid 图。
// 不依赖预处理器：读者本地 mdbook serve 不用多装任何东西。
(function () {
  function isDark() {
    var c = document.documentElement.className + " " + (document.body ? document.body.className : "");
    return /\b(navy|coal|ayu)\b/.test(c);
  }
  function render() {
    var blocks = document.querySelectorAll("pre > code.language-mermaid");
    if (!blocks.length || typeof mermaid === "undefined") return;
    blocks.forEach(function (code) {
      var pre = code.parentNode;
      var div = document.createElement("pre");
      div.className = "mermaid";
      div.textContent = code.textContent;
      pre.parentNode.replaceChild(div, pre);
    });
    mermaid.initialize({ startOnLoad: false, theme: isDark() ? "dark" : "default", securityLevel: "loose" });
    mermaid.run({ nodes: document.querySelectorAll("pre.mermaid") });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", render);
  else render();
})();
