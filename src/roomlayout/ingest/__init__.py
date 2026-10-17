"""Ingest stages: external captures -> contract objects (WS-E).

Each ingest module converts one external format at the boundary (ADR-0001) and imports
only ``roomlayout.schema``. ``roomlayout.ingest.roomplan`` reads Apple RoomPlan
``CapturedRoom`` JSON.
"""
