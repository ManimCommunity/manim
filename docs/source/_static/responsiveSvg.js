window.addEventListener("load", function () {
    const diagrams = document.querySelectorAll("object.inheritance.graphviz");
    const colorSchemeQuery = window.matchMedia("(prefers-color-scheme: dark)");

    function getDark() {
        // Furo's explicit theme takes precedence over the OS preference.
        let theme = document.body.dataset.theme;
        if (!theme) {
            try {
                theme = localStorage.getItem("theme");
            } catch {
                // Storage may be blocked; fall back to the OS preference.
            }
        }
        return theme === "dark" || (theme !== "light" && colorSchemeQuery.matches);
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
        for (const diagram of diagrams) {
            const svgRoot = diagram.contentDocument?.documentElement;
            if (!svgRoot) continue;

            let style = svgRoot.querySelector("style[data-manim-theme]");
            if (!style) {
                style = document.createElementNS("http://www.w3.org/2000/svg", "style");
                style.setAttribute("data-manim-theme", "");
                svgRoot.appendChild(style);
            }
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

    for (const diagram of diagrams) {
        diagram.addEventListener("load", setColorScheme);
    }
    setColorScheme();

    const observer = new MutationObserver(setColorScheme);
    observer.observe(document.body, {
        attributes: true,
        attributeFilter: ["data-theme"],
    });
    colorSchemeQuery.addEventListener("change", setColorScheme);
});
