"""Electronic-property result model (QMResult_elprop).

This module defines :class:`QMResult_elprop`, a Simstack/ODMantic model that
captures electronic properties (polarizabilities, hyperpolarizabilities,
etc.) associated with a quantum-chemistry calculation.

In the current transition phase, these fields also exist on
``QMResult``. ``QMResult_elprop`` provides a dedicated container so that
workflows and GUIs can treat electronic properties separately from the
core QM result. Over time, elprop-related fields and parsing logic can be
fully migrated here and removed from ``QMResult``.
"""

from __future__ import annotations

from typing import Optional, Dict, TYPE_CHECKING

import logging
import numpy as np
from odmantic import Model, Field

from simstack.models import simstack_model
from simstack.models.simple_table import SimpleTable
from simstack.util.ui_tools import ui_hide_fields

if TYPE_CHECKING:  # pragma: no cover - only for type checking, avoids cycles
    from .qm_result import QMResult
    from molecular_qm_orca.deprecated.orca_output import OrcaOutput


logger = logging.getLogger(__name__)


@simstack_model
class QMResult_elprop(Model):
    """Electronic properties / polarizabilities associated with a QMResult.

    The field set mirrors the elprop-related attributes currently present on
    :class:`QMResult`. Instances of this model can be attached to workflow
    results (e.g. ``node_runner.orca_elprop_result``) or queried directly
    from the database once persisted by the Simstack engine.
    """

    field_name: str = "QMResult_elprop"

    # Optional link back to the primary QMResult document as a plain string
    # (e.g. ``str(qm_result.id)``). This keeps the coupling light-weight and
    #avoids ODMantic-specific reference types at this stage.
    parent_qm_result_id: Optional[str] = Field(default=None)

    # ------------------------------------------------------------------
    # Hyperpolarizabilities and related aligned tensors (from ORCA %elprop)
    # ------------------------------------------------------------------
    static_hyperpolarizability_tensor: Optional[Dict[str, float]] = Field(
        default=None
    )
    aligned_static_hyperpolarizability_tensor: Optional[Dict[str, float]] = Field(
        default=None
    )
    x_aligned_static_hyperpolarizability_tensor: Optional[Dict[str, float]] = Field(
        default=None
    )
    pockels_hyperpolarizability_tensor: Optional[Dict[str, float]] = Field(
        default=None
    )
    aligned_pockels_hyperpolarizability_tensor: Optional[Dict[str, float]] = Field(
        default=None
    )
    frequency_doubling_hyperpolarizability_tensor: Optional[
        Dict[str, float]
    ] = Field(default=None)
    aligned_frequency_doubling_hyperpolarizability_tensor: Optional[
        Dict[str, float]
    ] = Field(default=None)

    # Tabular views for GUI display (1-row SimpleTables).
    aligned_static_hyperpolarizability_table: Optional[SimpleTable] = Field(
        default=None
    )
    x_aligned_static_hyperpolarizability_table: Optional[SimpleTable] = Field(
        default=None
    )
    aligned_pockels_hyperpolarizability_table: Optional[SimpleTable] = Field(
        default=None
    )
    aligned_frequency_doubling_hyperpolarizability_table: Optional[SimpleTable] = Field(
        default=None
    )

    # ------------------------------------------------------------------
    # Dipole/dipole static polarizability tensor and tables
    # ------------------------------------------------------------------
    dipole_polarizability_raw: Optional[Dict[str, float]] = Field(default=None)
    dipole_polarizability_diagonal: Optional[Dict[str, float]] = Field(
        default=None
    )
    dipole_polarizability_orientation: Optional[Dict[str, float]] = Field(
        default=None
    )
    dipole_polarizability_isotropic: Optional[Dict[str, float]] = Field(
        default=None
    )

    dipole_polarizability_raw_table: Optional[SimpleTable] = Field(default=None)
    dipole_polarizability_diagonal_table: Optional[SimpleTable] = Field(
        default=None
    )
    dipole_polarizability_orientation_table: Optional[SimpleTable] = Field(
        default=None
    )
    dipole_polarizability_isotropic_table: Optional[SimpleTable] = Field(
        default=None
    )

    # ------------------------------------------------------------------
    # Dipole/Quadrupole polarizabilities (including traceless variant)
    # ------------------------------------------------------------------
    dipole_quadrupole_polarizability_raw: Optional[Dict[str, float]] = Field(
        default=None
    )
    dipole_quadrupole_traceless_polarizability_raw: Optional[Dict[str, float]] = Field(
        default=None
    )

    dipole_quadrupole_polarizability_table: Optional[SimpleTable] = Field(
        default=None
    )
    dipole_quadrupole_traceless_polarizability_table: Optional[SimpleTable] = Field(
        default=None
    )

    # ------------------------------------------------------------------
    # Velocity polarizability tensor and tables
    # ------------------------------------------------------------------
    velocity_polarizability_raw: Optional[Dict[str, float]] = Field(default=None)
    velocity_polarizability_diagonal: Optional[Dict[str, float]] = Field(
        default=None
    )
    velocity_polarizability_orientation: Optional[Dict[str, float]] = Field(
        default=None
    )
    velocity_polarizability_isotropic: Optional[Dict[str, float]] = Field(
        default=None
    )

    velocity_polarizability_raw_table: Optional[SimpleTable] = Field(
        default=None
    )
    velocity_polarizability_diagonal_table: Optional[SimpleTable] = Field(
        default=None
    )
    velocity_polarizability_orientation_table: Optional[SimpleTable] = Field(
        default=None
    )
    velocity_polarizability_isotropic_table: Optional[SimpleTable] = Field(
        default=None
    )

    @classmethod
    def ui_base_schema(cls):
        """Base UI schema ordering for electronic properties.

        We mirror the layout used in ``QMResult.ui_base_schema`` but keep it
        focused entirely on electronic properties. Raw dict fields are hidden
        where table-based representations exist, so GUIs can show compact
        key/value tables by default while still having access to the
        underlying mappings if needed.
        """

        base_schema = {
            "ui:order": [
                # Hyperpolarizabilities (primarily static for ORCA)
                "static_hyperpolarizability_tensor",
                "aligned_static_hyperpolarizability_tensor",
                "aligned_static_hyperpolarizability_table",
                "x_aligned_static_hyperpolarizability_tensor",
                "x_aligned_static_hyperpolarizability_table",
                # Placeholders for extended backends (e.g. Turbomole)
                "pockels_hyperpolarizability_tensor",
                "aligned_pockels_hyperpolarizability_tensor",
                "aligned_pockels_hyperpolarizability_table",
                "frequency_doubling_hyperpolarizability_tensor",
                "aligned_frequency_doubling_hyperpolarizability_tensor",
                "aligned_frequency_doubling_hyperpolarizability_table",
                # Dipole polarizabilities
                "dipole_polarizability_raw",
                "dipole_polarizability_diagonal",
                "dipole_polarizability_orientation",
                "dipole_polarizability_isotropic",
                "dipole_polarizability_raw_table",
                "dipole_polarizability_diagonal_table",
                "dipole_polarizability_orientation_table",
                "dipole_polarizability_isotropic_table",
                # Dipole/Quadrupole polarizabilities
                "dipole_quadrupole_polarizability_raw",
                "dipole_quadrupole_traceless_polarizability_raw",
                "dipole_quadrupole_polarizability_table",
                "dipole_quadrupole_traceless_polarizability_table",
                # Velocity polarizabilities
                "velocity_polarizability_raw",
                "velocity_polarizability_diagonal",
                "velocity_polarizability_orientation",
                "velocity_polarizability_isotropic",
                "velocity_polarizability_raw_table",
                "velocity_polarizability_diagonal_table",
                "velocity_polarizability_orientation_table",
                "velocity_polarizability_isotropic_table",
                # Bookkeeping / linkage
                "parent_qm_result_id",
                "id",
            ],
        }

        hidden = [
            "static_hyperpolarizability_tensor",
            "x_aligned_static_hyperpolarizability_tensor",
            "aligned_static_hyperpolarizability_tensor",
            "pockels_hyperpolarizability_tensor",
            "frequency_doubling_hyperpolarizability_tensor",
            "dipole_polarizability_raw",
            "dipole_polarizability_diagonal",
            "dipole_polarizability_orientation",
            "dipole_polarizability_isotropic",
            "dipole_quadrupole_polarizability_raw",
            "dipole_quadrupole_traceless_polarizability_raw",
            "velocity_polarizability_raw",
            "velocity_polarizability_diagonal",
            "velocity_polarizability_orientation",
            "velocity_polarizability_isotropic",
        ]

        base_schema = ui_hide_fields(base_schema, hidden)
        return base_schema

    @classmethod
    def from_qm_result(cls, qm_result: "QMResult") -> "QMResult_elprop":
        """Construct a QMResult_elprop instance from an existing QMResult.

        This helper copies all known electronic-property fields from the
        provided ``qm_result``. It does **not** persist the instance; the
        caller (typically a node) is responsible for attaching it to a
        NodeRunner so that the Simstack engine can handle storage.
        """

        field_names = [
            # Hyperpolarizabilities
            "static_hyperpolarizability_tensor",
            "aligned_static_hyperpolarizability_tensor",
            "x_aligned_static_hyperpolarizability_tensor",
            "pockels_hyperpolarizability_tensor",
            "aligned_pockels_hyperpolarizability_tensor",
            "frequency_doubling_hyperpolarizability_tensor",
            "aligned_frequency_doubling_hyperpolarizability_tensor",
            "aligned_static_hyperpolarizability_table",
            "x_aligned_static_hyperpolarizability_table",
            "aligned_pockels_hyperpolarizability_table",
            "aligned_frequency_doubling_hyperpolarizability_table",
            # Dipole polarizabilities
            "dipole_polarizability_raw",
            "dipole_polarizability_diagonal",
            "dipole_polarizability_orientation",
            "dipole_polarizability_isotropic",
            "dipole_polarizability_raw_table",
            "dipole_polarizability_diagonal_table",
            "dipole_polarizability_orientation_table",
            "dipole_polarizability_isotropic_table",
            # Dipole/Quadrupole polarizabilities
            "dipole_quadrupole_polarizability_raw",
            "dipole_quadrupole_traceless_polarizability_raw",
            "dipole_quadrupole_polarizability_table",
            "dipole_quadrupole_traceless_polarizability_table",
            # Velocity polarizabilities
            "velocity_polarizability_raw",
            "velocity_polarizability_diagonal",
            "velocity_polarizability_orientation",
            "velocity_polarizability_isotropic",
            "velocity_polarizability_raw_table",
            "velocity_polarizability_diagonal_table",
            "velocity_polarizability_orientation_table",
            "velocity_polarizability_isotropic_table",
        ]

        data: Dict[str, object] = {}
        for name in field_names:
            if hasattr(qm_result, name):
                data[name] = getattr(qm_result, name)

        # Best-effort parent ID linking; qm_result.id might not be set until
        # persisted, so we guard this access.
        qm_id = getattr(qm_result, "id", None)
        if qm_id is not None:
            data["parent_qm_result_id"] = str(qm_id)

        return cls(**data)

    # ------------------------------------------------------------------
    # ORCA output parsing helper
    # ------------------------------------------------------------------
    @classmethod
    def from_orca_output(
        cls,
        orca_run: "OrcaOutput",
        parent_qm_result: Optional["QMResult"] = None,
        task_id: Optional[str] = None,
    ) -> "QMResult_elprop":
        """Build a QMResult_elprop directly from an OrcaOutput.

        This mirrors the elprop-related parts of
        ``QMResult.from_orca_output`` but keeps them in a dedicated
        container. ``parent_qm_result`` is optional and is only used to
        populate ``parent_qm_result_id`` and, as a fallback, to obtain
        the dipole moment vector when aligning the hyperpolarizability
        tensor.
        """

        # Local helper: dict -> SimpleTable with key/value layout
        def _dict_to_key_value_table(
            name: str, mapping: Optional[Dict[str, float]]
        ) -> Optional[SimpleTable]:
            if not mapping:
                return None
            try:
                table = SimpleTable(name=name)
                table.add_column("key", "string")
                table.add_column("value", "number")
                for k in sorted(mapping.keys()):
                    table.add_row({"key": k, "value": mapping[k]})
                return table
            except Exception as e_tbl:  # pragma: no cover - representational
                logger.warning(
                    "Failed to create SimpleTable %s from mapping: %s",
                    name,
                    e_tbl,
                )
                return None

        # Initialise all elprop fields to None; they will be filled
        # conditionally depending on what ORCA produced.
        data: Dict[str, object] = {
            # Hyperpolarizabilities
            "static_hyperpolarizability_tensor": None,
            "aligned_static_hyperpolarizability_tensor": None,
            "x_aligned_static_hyperpolarizability_tensor": None,
            "pockels_hyperpolarizability_tensor": None,
            "aligned_pockels_hyperpolarizability_tensor": None,
            "frequency_doubling_hyperpolarizability_tensor": None,
            "aligned_frequency_doubling_hyperpolarizability_tensor": None,
            # Hyperpol tables
            "aligned_static_hyperpolarizability_table": None,
            "x_aligned_static_hyperpolarizability_table": None,
            "aligned_pockels_hyperpolarizability_table": None,
            "aligned_frequency_doubling_hyperpolarizability_table": None,
            # Dipole polarizabilities (tensors + tables)
            "dipole_polarizability_raw": None,
            "dipole_polarizability_diagonal": None,
            "dipole_polarizability_orientation": None,
            "dipole_polarizability_isotropic": None,
            "dipole_polarizability_raw_table": None,
            "dipole_polarizability_diagonal_table": None,
            "dipole_polarizability_orientation_table": None,
            "dipole_polarizability_isotropic_table": None,
            # Dipole/Quadrupole polarizabilities
            "dipole_quadrupole_polarizability_raw": None,
            "dipole_quadrupole_traceless_polarizability_raw": None,
            "dipole_quadrupole_polarizability_table": None,
            "dipole_quadrupole_traceless_polarizability_table": None,
            # Velocity polarizabilities
            "velocity_polarizability_raw": None,
            "velocity_polarizability_diagonal": None,
            "velocity_polarizability_orientation": None,
            "velocity_polarizability_isotropic": None,
            "velocity_polarizability_raw_table": None,
            "velocity_polarizability_diagonal_table": None,
            "velocity_polarizability_orientation_table": None,
            "velocity_polarizability_isotropic_table": None,
        }

        try:
            # -----------------------------------------------------------------
            # Static hyperpolarizability tensor (ORCA)
            # -----------------------------------------------------------------
            try:
                hyperpol_info = getattr(orca_run, "_hyperpolarizability", None)
                static_tensor = None
                if isinstance(hyperpol_info, dict) and hyperpol_info:
                    static_tensor = dict(hyperpol_info)
                else:
                    logger.info(
                        "Reading elprop from task_id=%s --- "
                        "No _hyperpolarizability information on OrcaOutput",
                        task_id,
                    )

                data["static_hyperpolarizability_tensor"] = static_tensor
                logger.info(
                    "QMResult_elprop.static_hyperpolarizability_tensor = %s",
                    static_tensor,
                )

                # Dipole vector for alignment: prefer OrcaOutput attribute,
                # fall back to the parent QMResult if present.
                dipole_vec = getattr(orca_run, "dipole_moment", None)
                if dipole_vec is None and parent_qm_result is not None:
                    dipole_vec = getattr(parent_qm_result, "dipole_moment", None)

                logger.info("QMResult_elprop dipole_moment used for alignment: %s", dipole_vec)

                if isinstance(static_tensor, dict) and dipole_vec is not None:
                    logger.info("Creating aligned static hyperpolarizability tensors (z/x)")
                    from applications.electronic_structure.helper_libs.rotation_matrix_generation import (
                        align_tensor,
                    )

                    aligned_tensor = align_tensor(
                        hyperpol_tensor=static_tensor,
                        dipole_vector=dipole_vec,
                        logger=logger,
                    )
                    data["aligned_static_hyperpolarizability_tensor"] = aligned_tensor
                    logger.info(
                        "aligned_static_hyperpolarizability_tensor = %s",
                        aligned_tensor,
                    )

                    table = _dict_to_key_value_table(
                        "Aligned static hyperpolarizability", aligned_tensor
                    )
                    if table is not None:
                        data["aligned_static_hyperpolarizability_table"] = table
                        logger.info(
                            "aligned_static_hyperpolarizability_table created with %d rows",
                            len(table.rows),
                        )

                    x_aligned_tensor = align_tensor(
                        hyperpol_tensor=static_tensor,
                        dipole_vector=dipole_vec,
                        logger=logger,
                        axis="x",
                    )
                    data["x_aligned_static_hyperpolarizability_tensor"] = x_aligned_tensor
                    logger.info(
                        "x_aligned_static_hyperpolarizability_tensor = %s",
                        x_aligned_tensor,
                    )

                    x_table = _dict_to_key_value_table(
                        "X-aligned static hyperpolarizability", x_aligned_tensor
                    )
                    if x_table is not None:
                        data["x_aligned_static_hyperpolarizability_table"] = x_table
                        logger.info(
                            "x_aligned_static_hyperpolarizability_table created with %d rows",
                            len(x_table.rows),
                        )
                else:
                    logger.warning(
                        "Cannot create aligned static hyperpolarizability tensor because "
                        "tensor or dipole_moment is missing",
                    )
            except Exception as e_hyper:  # pragma: no cover - defensive logging
                logger.info(
                    "Reading elprop from task_id=%s --- Static hyperpolarizability parsing FAILED %s",
                    task_id,
                    e_hyper,
                )

            # -----------------------------------------------------------------
            # Static dipole polarizability tensor (ORCA %elprop Polar)
            # -----------------------------------------------------------------
            try:
                pol_info = getattr(orca_run, "_polarizability_dipole", None)
                if isinstance(pol_info, dict) and pol_info:
                    logger.info(
                        "Reading elprop from task_id=%s --- _polarizability_dipole keys: %s",
                        task_id,
                        sorted(pol_info.keys()),
                    )

                    def _matrix_to_dict(mat) -> Optional[Dict[str, float]]:
                        if mat is None:
                            return None
                        try:
                            arr = np.array(mat, dtype=float)
                        except Exception:
                            return None
                        if arr.shape != (3, 3):
                            return None
                        labels = ["x", "y", "z"]
                        out: Dict[str, float] = {}
                        for i, row in enumerate(labels):
                            for j, col in enumerate(labels):
                                out[row + col] = float(arr[i, j])
                        return out

                    def _vector_to_dict(vec) -> Optional[Dict[str, float]]:
                        if vec is None:
                            return None
                        try:
                            arr = np.array(vec, dtype=float).reshape(-1)
                        except Exception:
                            return None
                        if arr.shape[0] != 3:
                            return None
                        labels = ["x", "y", "z"]
                        return {axis: float(val) for axis, val in zip(labels, arr)}

                    raw_mat = pol_info.get("raw")
                    diag_vec = pol_info.get("diagonal")
                    orient_mat = pol_info.get("orientation")
                    iso_val = pol_info.get("isotropic")

                    data["dipole_polarizability_raw"] = _matrix_to_dict(raw_mat)
                    data["dipole_polarizability_diagonal"] = _vector_to_dict(diag_vec)
                    data["dipole_polarizability_orientation"] = _matrix_to_dict(orient_mat)

                    if iso_val is not None:
                        try:
                            iso_float = float(iso_val)
                            data["dipole_polarizability_isotropic"] = {
                                "isotropic": iso_float
                            }
                        except Exception:
                            logger.warning(
                                "Failed to cast isotropic polarizability value %r to float",
                                iso_val,
                            )

                    # Build key/value tables
                    data["dipole_polarizability_raw_table"] = _dict_to_key_value_table(
                        "Dipole polarizability (raw)",
                        data.get("dipole_polarizability_raw"),
                    )
                    data["dipole_polarizability_diagonal_table"] = _dict_to_key_value_table(
                        "Dipole polarizability (diagonal)",
                        data.get("dipole_polarizability_diagonal"),
                    )
                    data["dipole_polarizability_orientation_table"] = _dict_to_key_value_table(
                        "Dipole polarizability (orientation)",
                        data.get("dipole_polarizability_orientation"),
                    )
                    data["dipole_polarizability_isotropic_table"] = _dict_to_key_value_table(
                        "Dipole polarizability (isotropic)",
                        data.get("dipole_polarizability_isotropic"),
                    )
                else:
                    logger.info(
                        "Reading elprop from task_id=%s --- No _polarizability_dipole information",
                        task_id,
                    )
            except Exception as e_pol:  # pragma: no cover - defensive logging
                logger.info(
                    "Reading elprop from task_id=%s --- Dipole polarizability parsing FAILED %s",
                    task_id,
                    e_pol,
                )

            # -----------------------------------------------------------------
            # Mixed dipole/quadrupole static polarizability tensors
            # -----------------------------------------------------------------
            try:
                dq_info = getattr(orca_run, "_polarizability_dipole_quadrupole", None)
                dq_tr_info = getattr(
                    orca_run, "_polarizability_dipole_quadrupole_traceless", None
                )

                def _dq_matrix_to_dict(
                    raw: np.ndarray, comp_order: list[str]
                ) -> Optional[Dict[str, float]]:
                    if raw is None or comp_order is None:
                        return None
                    try:
                        arr = np.array(raw, dtype=float)
                    except Exception:
                        return None
                    if arr.shape != (3, 6):
                        return None

                    if len(comp_order) != 6:
                        comp_order_local = ["XX", "YY", "ZZ", "XY", "XZ", "YZ"]
                    else:
                        comp_order_local = comp_order

                    axes = ["x", "y", "z"]
                    quad_labels = [c.lower() for c in comp_order_local]
                    out: Dict[str, float] = {}
                    for i, ax in enumerate(axes):
                        for j, qlab in enumerate(quad_labels):
                            key = f"{ax}_{qlab}"
                            out[key] = float(arr[i, j])
                    return out

                if isinstance(dq_info, dict) and dq_info:
                    raw = dq_info.get("raw")
                    comp_order = dq_info.get("component_order")
                    data["dipole_quadrupole_polarizability_raw"] = _dq_matrix_to_dict(
                        raw, comp_order
                    )

                if isinstance(dq_tr_info, dict) and dq_tr_info:
                    raw_tr = dq_tr_info.get("raw")
                    comp_order_tr = dq_tr_info.get("component_order")
                    data["dipole_quadrupole_traceless_polarizability_raw"] = _dq_matrix_to_dict(
                        raw_tr, comp_order_tr
                    )

                if dq_info or dq_tr_info:
                    logger.info(
                        "Dipole/Quadrupole polarizabilities (raw/traceless) prepared for QMResult_elprop"
                    )
                else:
                    logger.info(
                        "Reading elprop from task_id=%s --- No dipole/quadrupole polarizability information",
                        task_id,
                    )

                # Build tables
                data["dipole_quadrupole_polarizability_table"] = _dict_to_key_value_table(
                    "Dipole/Quadrupole polarizability",
                    data.get("dipole_quadrupole_polarizability_raw"),
                )
                data[
                    "dipole_quadrupole_traceless_polarizability_table"
                ] = _dict_to_key_value_table(
                    "Dipole/Quadrupole traceless polarizability",
                    data.get("dipole_quadrupole_traceless_polarizability_raw"),
                )
            except Exception as e_dq:  # pragma: no cover - defensive logging
                logger.info(
                    "Reading elprop from task_id=%s --- Dipole/Quadrupole polarizability parsing FAILED %s",
                    task_id,
                    e_dq,
                )

            # -----------------------------------------------------------------
            # Velocity polarizability tensor
            # -----------------------------------------------------------------
            try:
                vel_info = getattr(orca_run, "_polarizability_velocity", None)
                if isinstance(vel_info, dict) and vel_info:
                    logger.info(
                        "Reading elprop from task_id=%s --- _polarizability_velocity keys: %s",
                        task_id,
                        sorted(vel_info.keys()),
                    )

                    def _vel_matrix_to_dict(mat) -> Optional[Dict[str, float]]:
                        if mat is None:
                            return None
                        try:
                            arr = np.array(mat, dtype=float)
                        except Exception:
                            return None
                        if arr.shape != (3, 3):
                            return None
                        labels = ["x", "y", "z"]
                        out: Dict[str, float] = {}
                        for i, row in enumerate(labels):
                            for j, col in enumerate(labels):
                                out[row + col] = float(arr[i, j])
                        return out

                    def _vel_vector_to_dict(vec) -> Optional[Dict[str, float]]:
                        if vec is None:
                            return None
                        try:
                            arr = np.array(vec, dtype=float).reshape(-1)
                        except Exception:
                            return None
                        if arr.shape[0] != 3:
                            return None
                        labels = ["x", "y", "z"]
                        return {axis: float(val) for axis, val in zip(labels, arr)}

                    raw_vel = vel_info.get("raw")
                    diag_vel = vel_info.get("diagonal")
                    orient_vel = vel_info.get("orientation")
                    iso_vel = vel_info.get("isotropic")

                    data["velocity_polarizability_raw"] = _vel_matrix_to_dict(raw_vel)
                    data["velocity_polarizability_diagonal"] = _vel_vector_to_dict(diag_vel)
                    data["velocity_polarizability_orientation"] = _vel_matrix_to_dict(
                        orient_vel
                    )

                    if iso_vel is not None:
                        try:
                            iso_float = float(iso_vel)
                            data["velocity_polarizability_isotropic"] = {
                                "isotropic": iso_float
                            }
                        except Exception:
                            logger.warning(
                                "Failed to cast velocity isotropic polarizability value %r to float",
                                iso_vel,
                            )

                    # Build tables
                    data["velocity_polarizability_raw_table"] = _dict_to_key_value_table(
                        "Velocity polarizability (raw)",
                        data.get("velocity_polarizability_raw"),
                    )
                    data[
                        "velocity_polarizability_diagonal_table"
                    ] = _dict_to_key_value_table(
                        "Velocity polarizability (diagonal)",
                        data.get("velocity_polarizability_diagonal"),
                    )
                    data[
                        "velocity_polarizability_orientation_table"
                    ] = _dict_to_key_value_table(
                        "Velocity polarizability (orientation)",
                        data.get("velocity_polarizability_orientation"),
                    )
                    data[
                        "velocity_polarizability_isotropic_table"
                    ] = _dict_to_key_value_table(
                        "Velocity polarizability (isotropic)",
                        data.get("velocity_polarizability_isotropic"),
                    )
                else:
                    logger.info(
                        "Reading elprop from task_id=%s --- No _polarizability_velocity information",
                        task_id,
                    )
            except Exception as e_vel:  # pragma: no cover - defensive logging
                logger.info(
                    "Reading elprop from task_id=%s --- Velocity polarizability parsing FAILED %s",
                    task_id,
                    e_vel,
                )

            logger.info(
                "Reading elprop from task_id=%s --- QMResult_elprop values prepared",
                task_id,
            )
        except Exception as e:
            logger.info(
                "Reading elprop from task_id=%s --- QMResult_elprop extraction FAILED %s",
                task_id,
                e,
            )

        # Link back to parent QMResult if available
        if parent_qm_result is not None:
            qm_id = getattr(parent_qm_result, "id", None)
            if qm_id is not None:
                data["parent_qm_result_id"] = str(qm_id)

        return cls(**data)
