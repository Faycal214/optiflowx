# Design System

This page documents the visual system used by OptiFlowX: color palette, typography, and component previews.

## Color Palette

Primary and semantic colors used across the site.

- Primary: `--md-primary-fg-color` (brand blue)
- Accent: `--md-accent-fg-color` (accent orange)
- Success: `--md-success-color`
- Warning: `--md-warning-color`
- Danger: `--md-danger-color`
- Info: `--md-info-color`

### Swatches

Primary

<div style={{display:'flex',gap:'12px',alignItems:'center'}}>
  <div style={{width:'140px',height:'48px',background:'var(--md-primary-fg-color)',borderRadius:'6px'}}></div>
  <div>Primary: <code>var(--md-primary-fg-color)</code></div>
</div>

Accent

<div style={{display:'flex',gap:'12px',alignItems:'center',marginTop:'0.6rem'}}>
  <div style={{width:'140px',height:'48px',background:'var(--md-accent-fg-color)',borderRadius:'6px'}}></div>
  <div>Accent: <code>var(--md-accent-fg-color)</code></div>
</div>

Semantic

<div style={{display:'flex',gap:'12px',flexWrap:'wrap',marginTop:'0.6rem'}}>
  <div style={{width:'120px',height:'40px',background:'var(--md-success-color)',borderRadius:'6px'}}>&nbsp;</div>
  <div style={{width:'120px',height:'40px',background:'var(--md-warning-color)',borderRadius:'6px'}}>&nbsp;</div>
  <div style={{width:'120px',height:'40px',background:'var(--md-danger-color)',borderRadius:'6px'}}>&nbsp;</div>
  <div style={{width:'120px',height:'40px',background:'var(--md-info-color)',borderRadius:'6px'}}>&nbsp;</div>
</div>

## Typography

- Text font: Inter (variable sizes)
- Code font: JetBrains Mono

### Heading scale

# H1 — Heading 1

## H2 — Heading 2

### H3 — Heading 3

#### H4 — Heading 4

##### H5 — Heading 5

###### H6 — Heading 6

Paragraph text sample: 

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum vestibulum.

Inline code example: `print("hello world")`

Code block sample:

```python
def add(a, b):
    return a + b

print(add(2, 3))
```

## Components

### Buttons

Primary button:

<button class="md-button md-button--primary">Primary</button>

Secondary button:

<button class="md-button md-button--secondary">Secondary</button>

### Admonitions

!!! note "Note"
    This is a note admonition.

!!! tip "Tip"
    This is a tip admonition.

!!! warning "Warning"
    This is a warning admonition.

!!! danger "Danger"
    This is a danger admonition.

### Tables

| Column A | Column B |
|---|---|
| Value 1 | Value 2 |
| Value 3 | Value 4 |



