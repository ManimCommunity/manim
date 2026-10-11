window.addEventListener("load", function () {
    const diagrams = document.querySelectorAll("object.inheritance.graphviz");

    // Determine the currently-active color scheme. Furo stores its theme choice
    // in localStorage["theme"] ("auto" | "dark" | "light") and reflects the
    // effective value on <body data-theme="...">. Prefer the real state over the
    // OS-level prefers-color-scheme so manual theme toggles are honoured.
    function getDark() {
        const bodyTheme =
            document.body && document.body.getAttribute("data-theme");
        if (bodyTheme === "dark") return true;
        if (bodyTheme === "light") return false;

        const stored = localStorage.getItem("theme");
        if (stored === "dark") return true;
        if (stored === "light") return false;

        // auto (or unset) -> follow the OS preference.
        return window.matchMedia("(prefers-color-scheme: dark)").matches;
    }

    function injectStyle(diagram) {
        const svgRoot =
            diagram.contentDocument && diagram.contentDocument.firstElementChild;
        if (!svgRoot) {
            return null;
        }
        const style = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "style"
        );
        svgRoot.appendChild(style);
        return style;
    }

    const styleElements = [];
    for (const diagram of diagrams) {
        let style = injectStyle(diagram);
        if (!style) {
            diagram.addEventListener("load", function () {
                style = injectStyle(diagram);
                if (style) {
                    styleElements.push(style);
                    setColorScheme();
                }
            });
        } else {
            styleElements.push(style);
        }
    }

    function setColorScheme() {
        let colors, additions = "";
        if (getDark()) {
            // Dark
            colors = {
                text: "#e07a5f",
                box: "#383838",
                edge: "#d0d0d0",
                background: "#131416"
            };
        } else {
            // Light
            colors = {
                text: "#e07a5f",
                box: "#fff",
                edge: "#413c3c",
                background: "#ffffff"
            };
            additions = `
            .node polygon {
                filter: drop-shadow(0 1px 3px #0002);
            }
            `;
        }
        for (const style of styleElements) {
            style.textContent = `
                svg {
                    background-color: ${colors.background};
                }

                .node text {
                    fill: ${colors.text};
                }

                .node polygon {
                    fill: ${colors.box};
                }

                .edge polygon {
                    fill: ${colors.edge};
                    stroke: ${colors.edge};
                }

                .edge path {
                    stroke: ${colors.edge};
                }
                ${additions}
            `;
        }
    }

    setColorScheme();

    // React to Furo's manual theme toggle (sets / removes data-theme on <body>).
    if (document.body) {
        const observer = new MutationObserver(setColorScheme);
        observer.observe(document.body, {
            attributes: true,
            attributeFilter: ["data-theme"],
        });
    }
    // React to OS-level auto-theme changes.
    window
        .matchMedia("(prefers-color-scheme: dark)")
        .addEventListener("change", setColorScheme);
});

