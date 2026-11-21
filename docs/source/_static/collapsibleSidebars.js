// Buttons that collapse the navigation sidebar and the table of contents on
// wide screens, where furo always shows both. The collapsed state is kept in
// localStorage and restored before the first paint by _templates/page.html.
(function () {
    // A panel with an arrow: the panel is on the side of the drawer the
    // button controls; the arrow points the way the drawer will move.
    const panel = (side) => {
        const x = side === "left" ? 9 : 15;
        return `<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M${x} 3v18"/>`;
    };
    const arrow = (points) => `<path d="${points}"/>`;
    const icon = (body, state) =>
        `<svg class="when-${state}" viewBox="0 0 24 24" fill="none" stroke="currentColor" ` +
        `stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;

    const items = [
        {
            name: "sidebar",
            label: "navigation sidebar",
            icons: icon(panel("left") + arrow("M16 15l-3-3 3-3"), "shown") +
                icon(panel("left") + arrow("M13 9l3 3-3 3"), "collapsed"),
        },
        {
            name: "toc",
            label: "table of contents",
            icons: icon(panel("right") + arrow("M8 9l3 3-3 3"), "shown") +
                icon(panel("right") + arrow("M11 15l-3-3 3-3"), "collapsed"),
        },
    ];
    const content = document.querySelector(".content");
    const icons = document.querySelector(".content-icon-container");
    const hasToc = !document.querySelector(".toc-drawer.no-toc");
    if (!content || !icons) return;

    for (const { name, label, icons: svg } of items) {
        if (name === "toc" && !hasToc) continue;

        const button = document.createElement("button");
        button.className = `drawer-toggle drawer-toggle-${name}`;
        button.innerHTML = svg;
        const update = () => {
            const collapsed = document.documentElement.classList.contains(`${name}-collapsed`);
            const text = `${collapsed ? "Show" : "Hide"} the ${label}`;
            button.title = text;
            button.setAttribute("aria-label", text);
            button.setAttribute("aria-expanded", String(!collapsed));
        };
        button.addEventListener("click", function () {
            const collapsed = document.documentElement.classList.toggle(`${name}-collapsed`);
            try {
                localStorage.setItem(`${name}-collapsed`, collapsed);
            } catch {
                // Storage may be blocked; the state then only lasts for this page.
            }
            update();
        });
        update();
        // The sidebar button sits in the content's left margin, next to the
        // sidebar; the contents button ends the row of page icons, where furo
        // puts its own contents button on narrower screens.
        if (name === "sidebar") {
            content.prepend(button);
        } else {
            icons.append(button);
        }
    }
})();
