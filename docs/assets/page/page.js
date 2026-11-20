(function () {
  "use strict";

  function $(sel) {
    return document.querySelector(sel);
  }

  var copyBtn = $("#copy-bib");
  var bibtex = $("#bibtex");

  if (copyBtn && bibtex && navigator.clipboard) {
    copyBtn.addEventListener("click", function () {
      navigator.clipboard.writeText(bibtex.textContent).then(
        function () {
          copyBtn.textContent = "Copied";
          setTimeout(function () {
            copyBtn.textContent = "Copy";
          }, 1600);
        },
        function () {
          copyBtn.textContent = "Select manually";
        }
      );
    });
  }
})();
