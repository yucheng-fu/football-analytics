import * as d3 from "d3";
import { FEATURE_LABELS } from "../config/featureLabels";
import type { FeatureContribution } from "../types";

const TOP_FEATURES = 8;
const WIDTH = 320;
const LABEL_WIDTH = 124;
const VALUE_WIDTH = 50;
const AXIS_HEIGHT = 22;
const MARKER_HEIGHT = 18;
const ROW_HEIGHT = 34;
const BAR_HEIGHT = 10;

const formatPp = d3.format("+.1f");
const formatPercent = d3.format(".1f");
const formatValue = d3.format(".3~g");

interface WaterfallRow {
  name: string;
  label: string;
  valueText: string;
  contribution: number;
  start: number;
  end: number;
}

function formatFeatureValue(feature: string, value: FeatureContribution["value"]) {
  if (value === null) {
    return "missing";
  }
  if (typeof value === "string") {
    return value;
  }
  if (feature === "under_pressure") {
    return value === 1 ? "Yes" : "No";
  }
  return formatValue(value);
}

function buildRows(baseValue: number, contributions: FeatureContribution[]): WaterfallRow[] {
  const top = contributions.slice(0, TOP_FEATURES);
  const rest = contributions.slice(TOP_FEATURES);
  const entries = top.map((item) => ({
    name: item.feature,
    label: FEATURE_LABELS[item.feature] ?? item.feature,
    valueText: formatFeatureValue(item.feature, item.value),
    contribution: item.contribution * 100,
  }));
  if (rest.length > 0) {
    entries.push({
      name: `${rest.length} other features`,
      label: `${rest.length} other features`,
      valueText: "combined",
      contribution: d3.sum(rest, (item) => item.contribution) * 100,
    });
  }

  let cumulative = baseValue * 100;
  return entries.map((entry) => {
    const start = cumulative;
    cumulative += entry.contribution;
    return { ...entry, start, end: cumulative };
  });
}

function truncateToWidth(textNode: SVGTextElement, maxWidth: number) {
  const fullText = textNode.textContent ?? "";
  let text = fullText;
  while (text.length > 1 && textNode.getComputedTextLength() > maxWidth) {
    text = text.slice(0, -1);
    textNode.textContent = `${text.trimEnd()}…`;
  }
}

export class ShapWaterfallRenderer {
  container: any;
  summary: any;
  svg: any;
  tooltip: any;

  constructor() {
    this.container = d3.select("#shap-explanation");
    this.summary = d3.select("#shap-summary");
    this.svg = d3.select("#shap-chart");
    this.tooltip = d3.select("#shap-tooltip");
  }

  hide() {
    this.container.property("hidden", true);
    this.tooltip.property("hidden", true);
    this.svg.selectAll("*").remove();
  }

  render(baseValue: number, contributions: FeatureContribution[]) {
    this.container.property("hidden", false);
    this.tooltip.property("hidden", true);

    const rows = buildRows(baseValue, contributions);
    const base = baseValue * 100;
    const final = rows.length > 0 ? rows[rows.length - 1].end : base;

    this.summary.html(
      `Probability of success for the average pass <strong>${formatPercent(base)}%</strong>`,
    );

    const rowsTop = AXIS_HEIGHT + MARKER_HEIGHT;
    const rowsHeight = rows.length * ROW_HEIGHT;
    const height = rowsTop + rowsHeight + MARKER_HEIGHT + 8;
    const plotLeft = LABEL_WIDTH;
    const plotRight = WIDTH - VALUE_WIDTH;

    const extent = d3.extent([base, ...rows.map((row) => row.end)]) as [number, number];
    const padding = Math.max((extent[1] - extent[0]) * 0.08, 0.5);
    const x = d3
      .scaleLinear()
      .domain([extent[0] - padding, extent[1] + padding])
      .range([plotLeft, plotRight])
      .nice(3)
      .clamp(true);
    const [domainMin, domainMax] = x.domain();
    x.domain([Math.max(domainMin, 0), Math.min(domainMax, 100)]);

    const svg = this.svg.attr("viewBox", `0 0 ${WIDTH} ${height}`).attr("height", height);
    svg.selectAll("*").remove();

    const axis = svg
      .append("g")
      .attr("class", "shap-axis")
      .attr("transform", `translate(0, ${AXIS_HEIGHT})`)
      .call(
        d3
          .axisTop(x)
          .ticks(3)
          .tickSize(-(rowsHeight + MARKER_HEIGHT * 2))
          .tickFormat((value: d3.NumberValue) => `${value}%`),
      );
    axis.select(".domain").remove();
    axis.selectAll(".tick text").attr("dy", "-0.2em");

    svg
      .append("text")
      .attr("class", "shap-unit")
      .attr("x", WIDTH - 6)
      .attr("y", AXIS_HEIGHT + 12)
      .attr("text-anchor", "end")
      .text("pp");

    svg
      .append("g")
      .attr("class", "shap-marker")
      .selectAll("text")
      .data([
        { value: base, y: AXIS_HEIGHT + 12, text: `Avg ${formatPercent(base)}%`, className: "" },
        { value: final, y: rowsTop + rowsHeight + 22, text: `This pass <tspan class="num">${formatPercent(final)}%</tspan>`, className: "final" },
      ])
      .join("text")
      .attr("class", (marker: { className: string }) => marker.className)
      .attr("x", (marker: { value: number }) => x(marker.value))
      .attr("y", (marker: { y: number }) => marker.y)
      .attr("text-anchor", (marker: { value: number }) => (x(marker.value) > plotRight - 40 ? "end" : "middle"))
      .html((marker: { text: string }) => marker.text);

    svg
      .append("circle")
      .attr("class", "shap-final-dot")
      .attr("cx", x(final))
      .attr("cy", rowsTop + rowsHeight + 6)
      .attr("r", 3);

    const connectorStops = [base, ...rows.map((row) => row.end)];
    svg
      .append("g")
      .attr("class", "shap-connectors")
      .selectAll("line")
      .data(connectorStops)
      .join("line")
      .attr("x1", (value: number) => x(value))
      .attr("x2", (value: number) => x(value))
      .attr("y1", (_value: number, index: number) =>
        index === 0 ? AXIS_HEIGHT + 15 : rowsTop + (index - 1) * ROW_HEIGHT + (ROW_HEIGHT + BAR_HEIGHT) / 2,
      )
      .attr("y2", (_value: number, index: number) =>
        index === rows.length ? rowsTop + rowsHeight + 4 : rowsTop + index * ROW_HEIGHT + (ROW_HEIGHT - BAR_HEIGHT) / 2,
      );

    const rowGroups = svg
      .append("g")
      .selectAll("g")
      .data(rows)
      .join("g")
      .attr("class", "shap-row")
      .attr("transform", (_row: WaterfallRow, index: number) => `translate(0, ${rowsTop + index * ROW_HEIGHT})`);

    rowGroups
      .append("rect")
      .attr("class", "wash")
      .attr("x", 0)
      .attr("width", WIDTH)
      .attr("height", ROW_HEIGHT)
      .attr("rx", 4);

    rowGroups
      .append("rect")
      .attr("class", (row: WaterfallRow) => (row.contribution >= 0 ? "bar pos" : "bar neg"))
      .attr("x", (row: WaterfallRow) => x(Math.min(row.start, row.end)))
      .attr("y", (ROW_HEIGHT - BAR_HEIGHT) / 2)
      .attr("width", (row: WaterfallRow) => Math.max(Math.abs(x(row.end) - x(row.start)), 1.5))
      .attr("height", BAR_HEIGHT)
      .attr("rx", 2);

    rowGroups
      .append("text")
      .attr("class", "label")
      .attr("x", 6)
      .attr("y", ROW_HEIGHT / 2 - 3)
      .text((row: WaterfallRow) => row.label)
      .each(function (this: SVGTextElement) {
        truncateToWidth(this, LABEL_WIDTH - 12);
      });

    rowGroups
      .append("text")
      .attr("class", "sub-label")
      .attr("x", 6)
      .attr("y", ROW_HEIGHT / 2 + 10)
      .text((row: WaterfallRow) => row.valueText)
      .each(function (this: SVGTextElement) {
        truncateToWidth(this, LABEL_WIDTH - 12);
      });

    rowGroups
      .append("text")
      .attr("class", "value")
      .attr("x", WIDTH - 6)
      .attr("y", ROW_HEIGHT / 2 + 4)
      .attr("text-anchor", "end")
      .text((row: WaterfallRow) => formatPp(row.contribution));

    rowGroups
      .append("title")
      .text(
        (row: WaterfallRow) =>
          `${row.name} = ${row.valueText}: ${formatPp(row.contribution)} pp (running total ${formatPercent(row.end)}%)`,
      );

    rowGroups
      .on("pointerenter", (event: PointerEvent, row: WaterfallRow) => this.showTooltip(event, row))
      .on("pointermove", (event: PointerEvent, row: WaterfallRow) => this.showTooltip(event, row))
      .on("pointerleave", () => this.tooltip.property("hidden", true));
  }

  private showTooltip(event: PointerEvent, row: WaterfallRow) {
    const containerNode = this.container.node() as HTMLElement;
    const bounds = containerNode.getBoundingClientRect();
    const direction = row.contribution >= 0 ? "raises" : "lowers";

    this.tooltip
      .property("hidden", false)
      .html(
        `<div class="tooltip-title">${row.label}</div>` +
          (row.name !== row.label ? `<div class="tooltip-muted">${row.name}</div>` : "") +
          `<div class="tooltip-row"><span>Value</span><span>${row.valueText}</span></div>` +
          `<div class="tooltip-row"><span>Effect</span><span class="${row.contribution >= 0 ? "pos" : "neg"}">${formatPp(row.contribution)} pp</span></div>` +
          `<div class="tooltip-row"><span>Running total</span><span>${formatPercent(row.end)}%</span></div>` +
          `<div class="tooltip-muted">${direction} the success probability</div>`,
      );

    const tooltipNode = this.tooltip.node() as HTMLElement;
    const left = Math.min(event.clientX - bounds.left + 12, bounds.width - tooltipNode.offsetWidth);
    const top = event.clientY - bounds.top - tooltipNode.offsetHeight - 10;
    this.tooltip.style("left", `${Math.max(left, 0)}px`).style("top", `${top}px`);
  }
}
