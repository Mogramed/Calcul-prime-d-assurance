from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from insurance_pricing.api.quote_store import StoredQuoteRecord


def build_quote_report_pdf(quote: StoredQuoteRecord) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setFillColor(colors.HexColor("#0E1A2A"))
    pdf.rect(0, height - 62 * mm, width, 62 * mm, fill=1, stroke=0)

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(18 * mm, height - 22 * mm, "Nova Assurances")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(18 * mm, height - 30 * mm, "Rapport de devis auto")
    pdf.drawString(18 * mm, height - 36 * mm, f"Reference devis : {quote.id}")
    pdf.drawString(
        18 * mm,
        height - 42 * mm,
        f"Date d'emission : {_format_datetime(quote.created_at_utc)}",
    )

    card_x = 18 * mm
    card_y = height - 100 * mm
    card_width = width - 36 * mm
    card_height = 38 * mm

    pdf.setFillColor(colors.HexColor("#F6EEE5"))
    pdf.roundRect(card_x, card_y, card_width, card_height, 8 * mm, fill=1, stroke=0)

    pdf.setFillColor(colors.HexColor("#A0622C"))
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(card_x + 8 * mm, card_y + 28 * mm, "PRIME ESTIMEE")
    pdf.setFillColor(colors.HexColor("#0E1A2A"))
    pdf.setFont("Helvetica-Bold", 28)
    pdf.drawString(
        card_x + 8 * mm,
        card_y + 15 * mm,
        _format_currency(quote.prime_prediction),
    )

    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(colors.HexColor("#4D5B6A"))
    pdf.drawString(
        card_x + 8 * mm,
        card_y + 8 * mm,
        "Estimation indicative, calculee a partir des informations renseignees.",
    )

    start_y = card_y - 12 * mm
    _draw_section(
        pdf,
        title="Formule",
        rows=[
            ("Formule", str(quote.input_payload.get("type_contrat", ""))),
            ("Paiement", str(quote.input_payload.get("freq_paiement", ""))),
            ("Usage", str(quote.input_payload.get("utilisation", ""))),
        ],
        top_y=start_y,
    )
    _draw_section(
        pdf,
        title="Conducteurs",
        rows=[
            ("Conducteur principal", f"{quote.input_payload.get('age_conducteur1', '')} ans"),
            ("Second conducteur", _second_driver_label(quote.input_payload)),
            ("Permis principal", f"{quote.input_payload.get('anciennete_permis1', '')} ans"),
        ],
        top_y=start_y - 42 * mm,
    )
    _draw_section(
        pdf,
        title="Vehicule",
        rows=[
            ("Marque", str(quote.input_payload.get("marque_vehicule", ""))),
            ("Modele", str(quote.input_payload.get("modele_vehicule", ""))),
            ("Motorisation", str(quote.input_payload.get("essence_vehicule", ""))),
            ("Valeur", f"{quote.input_payload.get('prix_vehicule', '')} EUR"),
        ],
        top_y=start_y - 84 * mm,
    )

    pdf.setStrokeColor(colors.HexColor("#D8C7B6"))
    pdf.line(18 * mm, 18 * mm, width - 18 * mm, 18 * mm)
    pdf.setFillColor(colors.HexColor("#6B7683"))
    pdf.setFont("Helvetica", 9)
    pdf.drawString(
        18 * mm,
        11 * mm,
        "Nova Assurances - rapport genere automatiquement pour consultation client.",
    )

    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _draw_section(
    pdf: canvas.Canvas,
    *,
    title: str,
    rows: list[tuple[str, str]],
    top_y: float,
) -> None:
    section_x = 18 * mm
    section_width = A4[0] - 36 * mm
    section_height = 34 * mm

    pdf.setFillColor(colors.white)
    pdf.setStrokeColor(colors.HexColor("#E6D9CB"))
    pdf.roundRect(section_x, top_y - section_height, section_width, section_height, 6 * mm, fill=1)
    pdf.setFillColor(colors.HexColor("#0E1A2A"))
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(section_x + 6 * mm, top_y - 8 * mm, title)

    cursor_y = top_y - 15 * mm
    pdf.setFont("Helvetica", 10)
    for label, value in rows:
        pdf.setFillColor(colors.HexColor("#6B7683"))
        pdf.drawString(section_x + 6 * mm, cursor_y, label)
        pdf.setFillColor(colors.HexColor("#0E1A2A"))
        pdf.drawRightString(section_x + section_width - 6 * mm, cursor_y, value or "Non renseigne")
        cursor_y -= 6 * mm


def _format_currency(value: float) -> str:
    return f"{value:,.2f} EUR".replace(",", " ").replace(".", ",")


def _format_datetime(value: datetime) -> str:
    return value.astimezone().strftime("%d/%m/%Y %H:%M")


def _second_driver_label(payload: dict[str, Any]) -> str:
    if payload.get("conducteur2") == "Yes":
        age = payload.get("age_conducteur2", "")
        return f"Oui ({age} ans)" if age else "Oui"
    return "Non"
