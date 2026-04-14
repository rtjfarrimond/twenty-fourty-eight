// Shared site navigation. Each page has `<div id="site-nav-root"></div>`
// as the first body element followed by `<script src="/nav.js"></script>`;
// this file injects the nav markup and marks the active link based on
// the current path.
(function () {
    const LINKS = [
        { href: '/',                              label: 'Game',       matches: (p) => p === '/' || p.endsWith('/index.html') },
        { href: '/dashboard.html',                label: 'Results',    matches: (p) => p === '/dashboard.html' || p === '/dashboard' },
        { href: '/dashboard/training.html',       label: 'Training',   matches: (p) => p.endsWith('/dashboard/training.html') },
        { href: '/dashboard/benchmarks.html',     label: 'Benchmarks', matches: (p) => p.endsWith('/dashboard/benchmarks.html') },
    ];

    function render() {
        const container = document.getElementById('site-nav-root');
        if (!container) return;

        const currentPath = window.location.pathname;
        const parts = [`<a href="/" class="nav-brand">2048</a>`];
        for (const link of LINKS) {
            const active = link.matches(currentPath) ? ' class="active"' : '';
            parts.push(`<a href="${link.href}"${active}>${link.label}</a>`);
        }
        container.outerHTML = `<nav class="site-nav">${parts.join('')}</nav>`;
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', render);
    } else {
        render();
    }
})();
