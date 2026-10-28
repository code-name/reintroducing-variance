/* Collapsible sidebar (all pages). Adapted from NSDA Corp's annotation apps (see NOTICE).
   State persists across pages via localStorage when available; the page works without it. */
(function () {
  "use strict";
  var KEY = "sidebar_collapsed";

  function read() { try { return localStorage.getItem(KEY) === "1"; } catch (e) { return false; } }
  function write(v) { try { localStorage.setItem(KEY, v ? "1" : "0"); } catch (e) { /* ignore */ } }

  function apply(collapsed) {
    var layout = document.querySelector(".layout");
    var btn = document.querySelector(".sidebar-toggle");
    if (!layout || !btn) return;
    layout.classList.toggle("sidebar-collapsed", collapsed);
    btn.textContent = collapsed ? "▶" : "◀";
    btn.title = collapsed ? "Expand sidebar" : "Collapse sidebar";
  }

  document.addEventListener("DOMContentLoaded", function () {
    apply(read());
    var btn = document.querySelector(".sidebar-toggle");
    if (btn) {
      btn.addEventListener("click", function () {
        var collapsed = !read();
        write(collapsed);
        apply(collapsed);
      });
    }
  });
})();
