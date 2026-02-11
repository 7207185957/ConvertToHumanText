(() => {
  const copyBtn = document.getElementById("copy-output");
  const output = document.getElementById("output_text");

  if (!copyBtn || !output) {
    return;
  }

  copyBtn.addEventListener("click", async () => {
    if (!output.value.trim()) {
      return;
    }
    try {
      await navigator.clipboard.writeText(output.value);
      const previous = copyBtn.textContent;
      copyBtn.textContent = "Copied";
      window.setTimeout(() => {
        copyBtn.textContent = previous;
      }, 1200);
    } catch (_err) {
      output.select();
      document.execCommand("copy");
    }
  });
})();
