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
        // Furo writes data-theme="auto" explicitly once the user has touched the
        // toggle; in that case (or when the attribute is absent entirely) the
        // effective scheme follows the OS preference.
        if (bodyTheme === "auto") {
            return window.matchMedia("(prefers-color-scheme: dark)").matches;
        }

        // data-theme not set yet (e.g. before Furo's own JS runs) — fall back
        // to the stored choice, then the OS preference. localStorage can throw
        // in Safari private mode or sandboxed iframes, so guard the read.
        let stored = null;
        try {
            stored = localStorage.getItem("theme");
        } catch (e) {
            // ignore SecurityError, fall through to matchMedia
        }
        if (stored === "dark") return true;
        if (stored === "light") return false;

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
    const pendingDiagrams = [];
    for (const diagram of diagrams) {
        let style = injectStyle(diagram);
        if (!style) {
            pendingDiagrams.push(diagram);
            diagram.addEventListener("load", function () {
                style = injectStyle(diagram);
                if (style) {
                    const idx = pendingDiagrams.indexOf(diagram);
                    if (idx !== -1) pendingDiagrams.splice(idx, 1);
                    styleElements.push(style);
                    setColorScheme();
                }
            });
        } else {
            styleElements.push(style);
        }
    }

    function setColorScheme() {
        // Retry injection for diagrams whose contentDocument was not ready at
        // window.load time (e.g. the object's load event already fired before
        // this script ran, so the listener attached above will never trigger).
        for (let i = pendingDiagrams.length - 1; i >= 0; i--) {
            const style = injectStyle(pendingDiagrams[i]);
            if (style) {
                styleElements.push(style);
                pendingDiagrams.splice(i, 1);
            }
        }

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
