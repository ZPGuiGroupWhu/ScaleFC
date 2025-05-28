import sys
sys.path.append(".")

from algorithm.util_tools import *
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.figure import Figure
from typing import Iterable, Literal, Optional, Union
from types import MappingProxyType
import itertools



__all__ = [
    "flow_get_color_generator",
    "flow_draw",
    "flow_draw_cluster_with_labels",
    "flow_draw_cluster_fp_fn_error_by_lines",
    "flow_draw_cluster_fp_fn_error_by_lines2",
    "get_figsize_and_axes_sizeinfo",
    "get_fig_and_axes_from_sizeinfo",
    "figsize_sizeinfo_vstack",
    "figsize_sizeinfo_hstack",
    "flow_zoom_out"
]

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ FLOW PLOT METHODS - START $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# functions of drawing flows

_my_color_map = MappingProxyType(
    {
        # Default
        "default": (
            'brown',
            'coral',
            # 'crimson',
            'darkgreen',
            'fuchsia',
            # 'gold',
            # 'green',
            'indigo',
            'khaki',
            'lightblue',
            'lightgreen',
            'pink',
            'plum',
            'purple',
            'sienna',
            'magenta',
            'navy',
            'olive',
            'tan',
            'orange',
            'teal',
            'maroon',
            'tomato',
            'turquoise',
            'violet',
            'wheat',
            'yellow',
            'yellowgreen',
            'blue',
            'red',
        ),
        "vintage": ('#0780cf', '#765005', '#fa6d1d', '#0e2c82', '#b6b51f', '#da1f18', '#701866', '#f47a75', '#009db2', '#024b51', '#0780cf', '#765005'),
        "feature": ('#63b2ee', '#76da91', '#f8cb7f', '#f89588', '#7cd6cf', '#9192ab', '#7898e1', '#efa666', '#eddd86', '#9987ce', '#63b2ee', '#76da91'),
        "gradient": ('#71ae46', '#96b744', '#c4cc38', '#ebe12a', '#eab026', '#e3852b', '#d85d2a', '#ce2626', '#ac2026', '#71ae46', '#96b744', '#c4cc38'),
      
        "fresh": ('#00a8e1', '#99cc00', '#e30039', '#fcd300', '#800080', '#00994e', '#ff6600', '#808000', '#db00c2', '#008080', '#0000ff', '#c8cc00'),
        
        "nostalgic": ('#3b6291', '#943c39', '#779043', '#624c7c', '#388498', '#bf7334', '#3f6899', '#9c403d', '#7d9847', '#675083', '#3b8ba1', '#c97937'),
        #
        "business": ('#194f97', '#555555', '#bd6b08', '#00686b', '#c82d31', '#625ba1', '#898989', '#9c9800', '#007f54', '#a195c5', '#103667', '#f19272'),
        
        "bright": ('#0e72cc', '#6ca30f', '#f59311', '#fa4343', '#16afcc', '#85c021', '#d12a6a', '#0e72cc', '#6ca30f', '#f59311', '#fa4343', '#16afcc'),
        
        "elegant": ('#3682be', '#45a776', '#f05326', '#eed777', '#334f65', '#b3974e', '#38cb7d', '#ddae33', '#844bb3', '#93c555', '#5f6694', '#df3881'),
        
        "soft": ('#5b9bd5', '#ed7d31', '#70ad47', '#ffc000', '#4472c4', '#91d024', '#b235e6', '#02ae75'),
        
        "subtle": ('#95a2ff', '#fa8080', '#ffc076', '#fae768', '#87e885', '#3cb9fc', '#73abf5', '#cb9bff', '#434348', '#90ed7d', '#f7a35c', '#8085e9'),
        
        "classic": ('#002c53', '#ffa510', '#0c84c6', '#ffbd66', '#f74d4d', '#2455a4', '#41b7ac'),
        
        "gorgeous": ('#fa2c7b', '#ff38e0', '#ffa235', '#04c5f3', '#0066fe', '#8932a5', '#c90444', '#cb9bff', '#434348', '#90ed7d', '#f7a35c', '#8085e9'),
        
        "technological": ('#05f8d6', '#0082fc', '#fdd845', '#22ed7c', '#09b0d3', '#1d27c9', '#f9e264', '#f47a75', '#009db2', '#024b51', '#0780cf', '#765005'),
        
        "vibrant1": ('#e75840', '#a565ef', '#628cee', '#eb9358', '#d05c7c', '#bb60b2', '#433e7c', '#f47a75', '#009db2', '#024b51', '#0780cf', '#765005'),
        
        "vibrant2": ('#ef4464', '#fad259', '#d22e8d', '#03dee0', '#d05c7c', '#bb60b2', '#433e7c', '#f47a75', '#009db2', '#024b51', '#0780cf', '#765005'),
        
        "minimal1": ('#929fff', '#9de0ff', '#ffa897', '#af87fe', '#7dc3fe', '#bb60b2', '#433e7c', '#f47a75', '#009db2', '#024b51', '#0780cf', '#765005'),
        
        "minimal2": ('#50c48f', '#26ccd8', '#3685fe', '#9977ef', '#f5616f', '#f7b13f', '#f9e264', '#f47a75', '#009db2', '#024b51', '#0780cf', '#765005'),
        "cool1": ('#bf19ff', '#854cff', '#5f45ff', '#02cdff', '#0090ff', '#314976', '#f9e264', '#f47a75', '#009db2', '#024b51', '#0780cf', '#765005'),
        "cool2": ('#45c8dc', '#854cff', '#5f45ff', '#47aee3', '#d5d6d8', '#96d7f9', '#f9e264', '#f47a75', '#009db2', '#024b51', '#0780cf', '#765005'),

        "warm": ('#9489fa', '#f06464', '#f7af59', '#f0da49', '#71c16f', '#2aaaef', '#5690dd', '#bd88f5', '#009db2', '#024b51', '#0780cf', '#765005'),
    }
)

_my_color_map_generator = MappingProxyType(
    {k: itertools.cycle(v) for k, v in _my_color_map.items()})


def flow_get_color_generator(
    name: Literal[
        'default',
        'vintage',
        'feature',
        'gradient',
        'fresh',
        'nostalgic',
        'business',
        'bright',
        'elegant',
        'soft',
        'subtle',
        'classic',
        'gorgeous',
        'technological',
        'vibrant1',
        'vibrant2',
        'minimal1',
        'minimal2',
        'cool1',
        'cool2',
        'warm',
    ],
    get_new: bool = True,
):
    assert name in _my_color_map, f"color_generator must be a string in {_my_color_map.keys()}"
    if get_new:
        return itertools.cycle(_my_color_map[name])
    else:
        return _my_color_map_generator[name]


# X's first column, second column, third column, and fourth column are ox, oy, dx, dy
def flow_draw(OD: np.ndarray, fig=None, ax=None, color: str = 'red', subplots_kwargs: Optional[dict] = None, **arrowprops_kwargs) -> tuple:
    """
    Draw flow visualization on a given figure and axis.

    Args:
        OD (np.ndarray): Array containing flow data. Each row represents a flow with four elements: [x1, y1, x2, y2].
        fig (matplotlib.figure.Figure, optional): Figure object to draw on. If not provided, a new figure will be created.
        ax (matplotlib.axes.Axes, optional): Axis object to draw on. If not provided, a new axis will be created.
        color (str, optional): Color of the flow arrows. Default is 'red'.
        subplots_kwargs (dict, optional): Keyword arguments for creating a new figure and axis.
        **arrowprops_kwargs: Keyword arguments for the arrow properties.

    Returns:
        fig (matplotlib.figure.Figure): The figure object used for drawing.
        ax (matplotlib.axes.Axes): The axis object used for drawing.
    """
    if hasattr(OD, 'to_array'):
        OD = OD.to_array()
    if np.ndim(OD) == 1:
        OD = [OD]

    if ax is None:
        if subplots_kwargs is None:
            subplots_kwargs = {}
        # Create a new figure and axis if not provided
        fig, ax = plt.subplots(**subplots_kwargs)

    mi = np.min(OD, axis=0)
    ma = np.max(OD, axis=0)

    mi_p = (min(mi[0], mi[2]), min(mi[1], mi[3]))
    ma_p = (max(ma[0], ma[2]), max(ma[1], ma[3]))
    ax.scatter(*mi_p, color='white', marker='o')
    ax.scatter(*ma_p, color='white', marker='o')
    for row in OD:
        # Add arrows
        ax.annotate('', xy=row[2:4], xytext=row[0:2], arrowprops=dict(
            arrowstyle='->', color=color, **arrowprops_kwargs))

    return fig, ax


def flow_draw_cluster_with_labels(
    flow_data: np.ndarray,
    labels: np.ndarray,
    color_generator: Union[
        Iterable,
        Literal[
            'default',
            'vintage',
            'feature',
            'gradient',
            'fresh',
            'nostalgic',
            'business',
            'bright',
            'elegant',
            'soft',
            'subtle',
            'classic',
            'gorgeous',
            'technological',
            'vibrant1',
            'vibrant2',
            'minimal1',
            'minimal2',
            'cool1',
            'cool2',
            'warm',
        ],
    ] = "feature",
    min_counts: int = 0,
    no_ticks: bool = True,
    title: str = '',
    new_color_gen: bool = True,
    noise_flow_color: str = 'lightgrey',
    draw_sequence: Literal["count", "index"] = 'index',
    fig=None,
    ax=None,
    subplots_kwargs: Optional[dict] = None,
    exclude_labels: Optional[Iterable] = None,
    **arrowprops_kwargs,
):
    """
    Draw flow clusters with labels, noise flows' color is grey.

    Args:
        flow_data (np.ndarray): Array of flow data.
        labels (np.ndarray): Array of labels corresponding to each flow data point.
        color_generator (Iterable): Iterator that generates colors for each cluster.
        min_counts (int): Minimum number of counts for a cluster to be considered.
        no_ticks (bool): Whether to show ticks on the axes.
        title (str): Title of the plot.
        new_color_gen (bool): Whether to create a new color generator.
        draw_sequence (Literal["count", "index"]): Whether to draw the clusters based on counts or indices.
        fig (matplotlib.figure.Figure, optional): Figure object to draw on. If not provided, a new figure will be created.
        ax (matplotlib.axes.Axes, optional): Axis object to draw on. If not provided, a new axis will be created.
        subplots_kwargs (dict, optional): Keyword arguments for creating a new figure and axis.
        exclude_labels (Iterable, optional): Labels to exclude from the plot.
        **arrowprops_kwargs: Keyword arguments for the arrow properties.


    Returns:
        fig: Figure object.
        ax: Axes object.
    """
    if exclude_labels is None:
        exclude_labels = []
    if new_color_gen:
        if isinstance(color_generator, str):
            assert color_generator in _my_color_map, f"color_generator must be a string in {_my_color_map.keys()}"
            _color_list = _my_color_map[color_generator]
            color_generator = itertools.cycle(_color_list)
        else:
            raise RuntimeError(
                "color_generator must be a string when new_default_color_gen is True.")

    elif isinstance(color_generator, str):
        assert color_generator in _my_color_map, f"color_generator must be a string in {_my_color_map.keys()}"
        color_generator = _my_color_map_generator[color_generator]

    if hasattr(flow_data, 'to_array'):
        flow_data = flow_data.to_array()
    assert len(flow_data) == len(
        labels), f"len(flow_data): {len(flow_data)}, len(labels): {len(labels)}"
    if ax is None:
        if subplots_kwargs is None:
            subplots_kwargs = {}
        fig, ax = plt.subplots(**subplots_kwargs)
    labels = np.asarray(labels, dtype=int)

    # draw noise flow
    if -1 in labels and -1 not in exclude_labels:
        noise_label = labels == -1
        noise_flow = flow_data[noise_label, :]
        flow_draw(noise_flow, ax=ax, fig=fig,
                  color=noise_flow_color, **arrowprops_kwargs)

    unique_elements, indices, counts = np.unique(
        labels, return_counts=True, return_index=True)
    if draw_sequence == "count":
        sa = np.argsort(counts)
        counts = counts[sa]
        unique_elements = unique_elements[sa]
    elif draw_sequence == "index":
        sa = np.argsort(indices)
        counts = counts[sa]
        unique_elements = unique_elements[sa]
    else:
        raise ValueError("draw_sequence must be 'count' or 'index'")

    index = range(len(counts))
    for idx in index:
        cur_label, c = unique_elements[idx], counts[idx]
        # print(f"cur_label:", cur_label)
        if cur_label < 0 or cur_label in exclude_labels:
            continue
        if c <= min_counts:
            curc = noise_flow_color
        else:
            curc = next(color_generator)
        cur_flow = flow_data[labels == cur_label, :]
        flow_draw(cur_flow, ax=ax, fig=fig, color=curc, **arrowprops_kwargs)

    if no_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title)

    return fig, ax


def flow_draw_cluster_fp_fn_error_by_lines(
    OD: np.ndarray,
    real_label: Union[np.ndarray, list, tuple],
    pred_label: Union[np.ndarray, list, tuple],
    *,
    draw_cluster_with_label_kwargs: Optional[dict] = None,
    legend_kwargs: dict = {"loc": 'best', "fancybox": True, "framealpha": 0.5},
    fn_line_kwargs: dict = {'color': 'red', 'label': 'FN Error', 'zorder': 9},
    fp_line_kwargs: dict = {'color': 'blue', 'label': 'FP Error', 'zorder': 9},
):
    real_label = np.asarray(real_label, dtype=int)
    pred_label = np.asarray(pred_label, dtype=int)

    # Avoid modifying original parameters
    fn_line_kwargs = fn_line_kwargs.copy()
    fp_line_kwargs = fp_line_kwargs.copy()

    # Find FN
    fn_idx = np.where((real_label != -1) & (pred_label == -1))[0]
    fp_idx = np.where((real_label == -1) & (pred_label != -1))[0]

    n_pred_lab = pred_label.copy()
    if draw_cluster_with_label_kwargs is None:
        draw_cluster_with_label_kwargs = {}

    # Use trick to assign -2 and -3
    if fn_idx.size > 0:
        n_pred_lab[fn_idx] = -2
    if fp_idx.size > 0:
        n_pred_lab[fp_idx] = -3

    # Then draw
    fig, ax = flow_draw_cluster_with_labels(
        OD, n_pred_lab, **draw_cluster_with_label_kwargs)

    # Then extract flows to draw
    line_x, line_y = OD[0][0:2]
    if fn_idx.size > 0:
        # First draw a line
        OD_FN = OD[fn_idx, :]
        ax.plot([line_x], [line_y], **fn_line_kwargs)
        fn_line_kwargs.pop('label')

        # Then draw flows
        flow_draw(OD_FN, fig=fig, ax=ax, **fn_line_kwargs)

    if fp_idx.size > 0:
        OD_FP = OD[fp_idx, :]
        ax.plot([line_x], [line_y], **fp_line_kwargs)
        # ax.scatter([line_xy], [line_xy], **fp_line_kwargs)

        fp_line_kwargs.pop('label')
        flow_draw(OD_FP, fig=fig, ax=ax, **fp_line_kwargs)

    if fn_idx.size > 0 or fp_idx.size > 0:
        ax.legend(**legend_kwargs)


def flow_draw_cluster_fp_fn_error_by_lines2(
    OD: np.ndarray,
    real_label: Union[np.ndarray, list, tuple],
    pred_label: Union[np.ndarray, list, tuple],
    *,
    draw_cluster_with_label_kwargs: Optional[dict] = None,
    fn_line_kwargs: dict = {'color': 'red', 'zorder': 9, "lw": 0.5},
    fp_line_kwargs: dict = {'color': 'blue', 'zorder': 9, "lw": 0.5},
    draw_inset: bool = False
):
    real_label = np.asarray(real_label, dtype=int)
    pred_label = np.asarray(pred_label, dtype=int)

    # Avoid modifying original parameters
    fn_line_kwargs = fn_line_kwargs.copy()
    fp_line_kwargs = fp_line_kwargs.copy()

    # Find FN
    fn_idx = np.where((real_label != -1) & (pred_label == -1))[0]
    fp_idx = np.where((real_label == -1) & (pred_label != -1))[0]

    if draw_cluster_with_label_kwargs is None:
        draw_cluster_with_label_kwargs = {}

    # Then draw all flows
    fig, ax = flow_draw_cluster_with_labels(
        OD, pred_label, **draw_cluster_with_label_kwargs)

    # Then extract flows to draw
    if fn_idx.size > 0:
        # First draw a line
        OD_FN = OD[fn_idx, :]

        # Then draw flows
        # flow_draw(OD_FN, fig=fig, ax=ax, **fn_line_kwargs)
        newOD2 = flow_zoom_out(OD_FN, 0.7)
        # Extract start and end coordinates
        x_coords = newOD2[:, [0, 2]]
        y_coords = newOD2[:, [1, 3]]

        # Create a plot
        # Use batch plotting method to draw all line segments
        ax.plot(x_coords.T, y_coords.T, **fn_line_kwargs)

    if fp_idx.size > 0:
        OD_FP = OD[fp_idx, :]
        newOD2 = flow_zoom_out(OD_FP, 0.7)
        # Extract start and end coordinates
        x_coords = newOD2[:, [0, 2]]
        y_coords = newOD2[:, [1, 3]]

        # Create a plot
        # Use batch plotting method to draw all line segments
        ax.plot(x_coords.T, y_coords.T, **fp_line_kwargs)

    # Draw inset
    if draw_inset:
        # Draw C1 and C2
        curgen = itertools.cycle(_my_color_map["feature"])
        new_ax = inset_axes(ax, width="30%", height="30%", loc='lower left',
                            bbox_to_anchor=(-0.01, -0.01, 1, 1), bbox_transform=ax.transAxes)
        indices = np.where((real_label == 0) | (real_label == 1))[0]
        # Find FP and FN
        newOD = OD[indices]
        fp_fn_lst = fp_idx.tolist() + fn_idx.tolist()
        indices = indices.tolist()
        left, bottom, right, top = flow_envelope(newOD)
        for idx in fp_fn_lst:
            ox, oy, dx, dy = OD[idx]
            nx = (ox+dx) / 2
            ny = (oy+dy) / 2
            if left <= nx <= right and bottom <= ny <= top:
                indices.append(idx)

        indices = np.asarray(indices, dtype=int)

        newOD = OD[indices]
        new_reallabel = real_label[indices]
        new_predlabel = pred_label[indices]
        draw_cluster_with_label_kwargs["ax"] = new_ax
        draw_cluster_with_label_kwargs["new_color_gen"] = False
        draw_cluster_with_label_kwargs["color_generator"] = curgen

        flow_draw_cluster_fp_fn_error_by_lines2(newOD, new_reallabel, new_predlabel, draw_cluster_with_label_kwargs=draw_cluster_with_label_kwargs,
                                                fn_line_kwargs=fn_line_kwargs, fp_line_kwargs=fp_line_kwargs, draw_inset=False)
        for spine in new_ax.spines.values():
            spine.set_linewidth(0.5)  # Thickness, adjust as needed
            pass
        new_ax.patch.set_alpha(0.5)  # Set transparency

        count_0_1 = np.unique(new_predlabel).size

        # Do it again
        curgen = itertools.cycle(_my_color_map["feature"])
        for _ in range(count_0_1):
            next(curgen)
        new_ax = inset_axes(ax, width="30%", height="30%", loc='center right', bbox_to_anchor=(
            0.01, 0.05, 1, 1), bbox_transform=ax.transAxes)
        indices = np.where((real_label == 2) | (real_label == 3))[0]
        # Find FP and FN
        newOD = OD[indices]
        fp_fn_lst = fp_idx.tolist() + fn_idx.tolist()
        indices = indices.tolist()
        left, bottom, right, top = flow_envelope(newOD)
        for idx in fp_fn_lst:
            ox, oy, dx, dy = OD[idx]
            nx = (ox+dx) / 2
            ny = (oy+dy) / 2
            if left <= nx <= right and bottom <= ny <= top:
                indices.append(idx)

        indices = np.asarray(indices)

        newOD = OD[indices]
        new_reallabel = real_label[indices]
        new_predlabel = pred_label[indices]
        draw_cluster_with_label_kwargs["ax"] = new_ax
        draw_cluster_with_label_kwargs["new_color_gen"] = False
        draw_cluster_with_label_kwargs["color_generator"] = curgen

        flow_draw_cluster_fp_fn_error_by_lines2(newOD, new_reallabel, new_predlabel, draw_cluster_with_label_kwargs=draw_cluster_with_label_kwargs,
                                                fn_line_kwargs=fn_line_kwargs, fp_line_kwargs=fp_line_kwargs, draw_inset=False)
        for spine in new_ax.spines.values():
            spine.set_linewidth(0.5)  # Thickness, adjust as needed
            pass
        new_ax.patch.set_alpha(0.5)  # Set transparency

# ******************************************* FLOW PLOT METHODS - END *******************************************
# Return the figsize and size information for each axes, i.e., x, y, width, height


def get_figsize_and_axes_sizeinfo(row_num=4, col_num=6, A4_width=6.05, height_width_ratio=1,
                                  left_margin=0.1,
                                  right_margin=0.1,
                                  bottom_margin=0.1,
                                  top_margin=0.15,
                                  twofigs_width_margin=0.05,
                                  twofigs_height_margin=0.05) -> tuple[tuple[float, float], np.ndarray[tuple[float, float, float, float]]]:
    """return (figsize, size_info), figsize (tuple): (width, height), size_info(np.ndarray): (x, y, width, height)"""
    # The edge of the entire large image
    # Unit inches
    width_size = (A4_width - right_margin - left_margin -
                  twofigs_width_margin * (col_num - 1)) / col_num
    height_size = width_size * height_width_ratio
    # print(sf_size)

    figsize = (left_margin + right_margin + col_num * width_size + (col_num - 1) * twofigs_width_margin,
               row_num * height_size + bottom_margin + top_margin + (row_num - 1) * twofigs_height_margin)

    # 根据0,0计算出坐标的大小
    # Calculate coordinate sizes based on 0,0
    def func(a, b): return ((left_margin + b * (width_size + twofigs_width_margin))/figsize[0],
                            (bottom_margin + (row_num - 1 - a) *
                             (height_size + twofigs_height_margin)) / figsize[1],
                            width_size / figsize[0],
                            height_size / figsize[1])

    # Create array
    # print(figsize)
    size_info = np.empty((row_num, col_num), dtype=object)
    for a in range(row_num):
        for b in range(col_num):
            size_info[a, b] = func(a, b)

    return (figsize, size_info.flat)


def get_fig_and_axes_from_sizeinfo(figsize, *sizeinfo: np.ndarray) -> tuple[Figure, list[Axes]]:
    fig = plt.figure(figsize=figsize)
    axes = []
    flattened_arrays = np.concatenate([np.ravel(arr) for arr in sizeinfo])
    for x in flattened_arrays:
        # print(x)
        cur_ax = fig.add_axes(x)
        axes.append(cur_ax)
        for spine in cur_ax.spines.values():
            spine.set_linewidth(0.5)  # Thickness can be adjusted as needed

    return (fig, axes)


# Returns figsize and sizeinfo for each axis
def figsize_sizeinfo_vstack(figsize1, figsize2, axs1_sizeinfo: np.ndarray, axs2_sizeinfo: np.ndarray, merge_ax=False) -> tuple[tuple[float, float], np.ndarray[tuple], np.ndarray[tuple]]:
    fig1_width, fig1_height = figsize1
    fig2_width, fig2_height = figsize2
    total_width = max(fig1_width, fig2_width)
    total_height = fig1_height + fig2_height

    axs1_sizeinfo = np.copy(axs1_sizeinfo)
    axs2_sizeinfo = np.copy(axs2_sizeinfo)

    if np.ndim(axs1_sizeinfo) == 1:
        axs1_sizeinfo = axs1_sizeinfo.reshape(1, -1)
    N, M = axs1_sizeinfo.shape
    for i in range(N):
        for j in range(M):
            x, y, w, h = axs1_sizeinfo[i, j]
            x = x * fig1_width / total_width
            y = (y * fig1_height + fig2_height) / total_height
            w = w * fig1_width / total_width
            h = h * fig1_height / total_height
            axs1_sizeinfo[i, j] = (x, y, w, h)

    if np.ndim(axs2_sizeinfo) == 1:
        axs2_sizeinfo = axs2_sizeinfo.reshape(1, -1)
    N, M = axs2_sizeinfo.shape
    for i in range(N):
        for j in range(M):
            x, y, w, h = axs2_sizeinfo[i, j]
            x = x * fig2_width / total_width
            y = y * fig2_height / total_height
            w = w * fig2_width / total_width
            h = h * fig2_height / total_height
            axs2_sizeinfo[i, j] = (x, y, w, h)

    if merge_ax:
        axs1_sizeinfo = np.concatenate(
            [axs1_sizeinfo.flat, axs2_sizeinfo.flat])
        return ((total_width, total_height), axs1_sizeinfo)
    return ((total_width, total_height), axs1_sizeinfo, axs2_sizeinfo)


def figsize_sizeinfo_hstack(figsize1, figsize2, axs1_sizeinfo: np.ndarray, axs2_sizeinfo: np.ndarray) -> tuple[tuple[float, float], np.ndarray[tuple], np.ndarray[tuple]]:
    fig1_width, fig1_height = figsize1
    fig2_width, fig2_height = figsize2
    total_width = fig1_width + fig2_width
    total_height = max(fig1_height, fig2_height)

    N, M = axs1_sizeinfo.shape
    for i in range(N):
        for j in range(M):
            x, y, w, h = axs1_sizeinfo[i, j]
            x = x * fig1_width / total_width
            y = y * fig1_height / total_height
            w = w * fig1_width / total_width
            h = h * fig1_height / total_height
            axs1_sizeinfo[i, j] = (x, y, w, h)

    N, M = axs2_sizeinfo.shape
    for i in range(N):
        for j in range(M):
            x, y, w, h = axs2_sizeinfo[i, j]
            x = (x * fig2_width + fig1_width) / total_width
            y = y * fig2_height / total_height
            w = w * fig2_width / total_width
            h = h * fig2_height / total_height
            axs2_sizeinfo[i, j] = (x, y, w, h)
    return ((total_width, total_height), axs1_sizeinfo, axs2_sizeinfo)


# 往两边延长
# Extend in both directions
def flow_extension(OD: np.ndarray, od_ex_len: float, do_ex_len: float, copy=True) -> np.ndarray:
    if copy:
        OD = np.copy(OD)
    ox = OD[:, 0]
    oy = OD[:, 1]
    dx = OD[:, 2]
    dy = OD[:, 3]

    ang = flow_angle(OD)
    dx += od_ex_len * np.cos(ang)
    dy += od_ex_len * np.sin(ang)

    ox -= do_ex_len * np.cos(ang)
    oy -= do_ex_len * np.sin(ang)

    OD[:, 0] = ox
    OD[:, 1] = oy
    OD[:, 2] = dx
    OD[:, 3] = dy

    return OD


def flow_zoom_out(OD: np.ndarray, ratio: float = 0.1, copy=True) -> np.ndarray:

    assert 0 <= ratio < 1  # Ensure non-negative zoom-out ratio
    if ratio == 0:
        return OD
    # Compute the scaled length for zooming
    l = flow_length(OD) * ratio * (-0.5)

    # Perform the zoom-out operation on the OD flows
    return flow_extension(OD, l, l, copy)


def flow_envelope(OD: np.ndarray):
    mi = np.min(OD, axis=0)
    ma = np.max(OD, axis=0)
    left, bottom = min(mi[0], mi[2]), min(mi[1], mi[3])
    right, top = (max(ma[0], ma[2]), max(ma[1], ma[3]))
    return left, bottom, right, top
