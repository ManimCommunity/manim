(function () {
    const items = [
        { name: "sidebar", icon: "svg-menu", label: "Toggle navigation sidebar", position: "before" },
        { name: "toc", icon: "svg-toc", label: "Toggle table of contents", position: "after" },
    ];
    const container = document.querySelector(".content-icon-container");
    const hasToc = !document.querySelector(".toc-drawer.no-toc");

    for (const { name, icon, label, position } of items) {
        if (name === "toc" && !hasToc) continue;

        const button = document.createElement("button");
        button.className = `drawer-toggle drawer-toggle-${name}`;
        button.title = label;
        button.setAttribute("aria-label", label);
        button.innerHTML = `<svg><use href="#${icon}"></use></svg>`;
        button.addEventListener("click", function () {
            const collapsed = document.documentElement.classList.toggle(`${name}-collapsed`);
            try {
                localStorage.setItem(`${name}-collapsed`, collapsed);
            } catch {
                // Storage may be blocked; the state then only lasts for this page.
            }
        });
        // Place each toggle on the side of the drawer it controls.
        if (position === "before") {
            container.before(button);
        } else if (position === "after") {
            container.append(button)
        }
    }
})();
