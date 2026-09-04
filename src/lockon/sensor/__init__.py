"""lockon.sensor — see plan.md §9 D1 for the import rule."""

from lockon.sensor.registry import RENDERERS, RenderContext, register
from lockon.sensor.sensor import Sensor, side_by_side

__all__ = ["RENDERERS", "RenderContext", "Sensor", "register", "side_by_side"]

