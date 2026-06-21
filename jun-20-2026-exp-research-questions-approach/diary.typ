#set document(
  title: "Redoing Experiments with Research Questions Mindset",
  author: "Ana Clara Zoppi Serpa",
  date: datetime(year: 2026, month: 6, day: 21),
)
#set text(font: "New Computer Modern")
#set heading(numbering: "1.1")
#show cite: set text(fill: rgb("#2563eb"))

#let title-page(doc) = {
  align(center + horizon)[
    #text(size: 22pt, weight: "bold")[#doc.title]
    #v(1em)
    #text(size: 14pt)[#doc.author.join(", ")]
    #v(0.5em)
    #text(size: 11pt, fill: gray)[#doc.date.display("[month repr:long] [year]")]
  ]
}

#context title-page(document)
#pagebreak()

#outline(indent: auto, title: "Table of Contents")
#pagebreak()

= Brainstorm

= Research Questions

