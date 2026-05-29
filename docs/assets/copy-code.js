(function () {
  function getCodeText(code) {
    return code.innerText.replace(/\n$/, "");
  }

  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }

    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.top = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }

  function addCopyButton(pre, code) {
    if (pre.dataset.copyReady === "true") {
      return;
    }

    const text = getCodeText(code);
    if (!text.trim()) {
      return;
    }

    pre.dataset.copyReady = "true";

    const frame = document.createElement("div");
    frame.className = "code-copy-frame";
    pre.parentNode.insertBefore(frame, pre);
    frame.appendChild(pre);

    const button = document.createElement("button");
    button.type = "button";
    button.className = "code-copy-button";
    button.setAttribute("aria-label", "Copy code");
    button.title = "Copy code";
    button.textContent = "Copy";

    let resetTimer = null;
    button.addEventListener("click", async () => {
      window.clearTimeout(resetTimer);
      try {
        await copyText(getCodeText(code));
        button.textContent = "Copied";
      } catch (_error) {
        button.textContent = "Failed";
      }
      resetTimer = window.setTimeout(() => {
        button.textContent = "Copy";
      }, 1600);
    });

    frame.appendChild(button);
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("pre > code").forEach((code) => {
      addCopyButton(code.parentElement, code);
    });
  });
})();
