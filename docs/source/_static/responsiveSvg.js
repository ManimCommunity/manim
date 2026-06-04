
window.addEventListener("load", function () {
    const styleElements = []
    const colorSchemeQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const diagrams = document.querySelectorAll("object.inheritance.graphviz");

    for (let diagram of diagrams) {
        style = document.createElement('style');
        styleElements.push(style);
        console.log(diagram);
        diagram.contentDocument.firstElementChild.appendChild(style);
    }

    function setColorScheme(e) {
        let colors, additions = "";
        if (e.matches) {
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
            `
        }
        for (let style of styleElements) {
            style.innerHTML = `
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

    setColorScheme(colorSchemeQuery);
    colorSchemeQuery.addEventListener("change", setColorScheme);
});
