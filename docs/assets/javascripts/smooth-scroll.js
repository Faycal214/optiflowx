// Smooth scrolling for anchor links and TOC active highlighting
(function() {
  function offsetScrollTo(element) {
    if (!element) return;
    var headerOffset = 84; // estimated header height
    var elementPosition = element.getBoundingClientRect().top + window.pageYOffset;
    var offsetPosition = elementPosition - headerOffset;
    window.scrollTo({ top: offsetPosition, behavior: 'smooth' });
  }

  document.addEventListener('click', function(e) {
    var anchor = e.target.closest('a[href^="#"]');
    if (!anchor) return;
    var href = anchor.getAttribute('href');
    if (href === '#' || href === '#!') return;
    var id = href.slice(1);
    var target = document.getElementById(id);
    if (target) {
      e.preventDefault();
      offsetScrollTo(target);
      history.replaceState(null, '', '#' + id);
    }
  }, { passive: false });

  // Highlight TOC entry on scroll
  var tocLinks = Array.from(document.querySelectorAll('.md-nav--toc a.md-nav__link'));
  var headings = tocLinks.map(l => document.getElementById(l.getAttribute('href')?.slice(1))).filter(Boolean);
  function onScroll() {
    var fromTop = window.scrollY + 96;
    var currentId = null;
    for (var i = 0; i < headings.length; i++) {
      var h = headings[i];
      if (h.offsetTop <= fromTop) currentId = h.id;
    }
    tocLinks.forEach(function(link){
      var linkId = link.getAttribute('href')?.slice(1);
      if (linkId === currentId) link.classList.add('md-nav__link--active'); else link.classList.remove('md-nav__link--active');
    });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('load', onScroll);
})();
