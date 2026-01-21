"""
Automated SSM Presentation Builder
Creates all necessary files and compiles the Beamer presentation.

Usage:
    python build_presentation.py

Output:
    ./presentation_output/ssm_presentation.pdf
"""

import os
import shutil
import subprocess
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("presentation_output")
LATEX_FILE = "ssm_presentation.tex"

# LaTeX content
LATEX_CONTENT = r"""\documentclass[aspectratio=169]{beamer}

% Theme and Color Setup
\usetheme{Madrid}
\usecolortheme{beaver}
\usefonttheme{professionalfonts}

% Packages
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{shapes, arrows, positioning}
\usepackage{booktabs}
\usepackage{hyperref}

% Custom Colors based on code context (Mamba-ish colors)
\definecolor{mambapurple}{RGB}{128, 0, 128}
\definecolor{codebg}{RGB}{240, 240, 240}
% Consistent color scheme from visualizations
\definecolor{inputblue}{RGB}{46, 134, 171}
\definecolor{statepurple}{RGB}{162, 59, 114}
\definecolor{outputgreen}{RGB}{6, 167, 125}
\definecolor{patchorange}{RGB}{241, 143, 1}
\definecolor{highlight}{RGB}{199, 62, 29}

% Title Information
\title[SSM Time Series]{State Space Models for Time Series}
\subtitle{Architecture, Mathematics, and Implementation}
\author{Leandro Stival}
\institute[WUR]{Wageningen University \& Research}
\date{\today}

\begin{document}

% -----------------------------------------------------------------------------
% Slide 1: Title
% -----------------------------------------------------------------------------
\begin{frame}
    \titlepage
\end{frame}

% -----------------------------------------------------------------------------
% Slide 2: Table of Contents
% -----------------------------------------------------------------------------
\begin{frame}{Overview}
    \tableofcontents
\end{frame}

% =============================================================================
% Section 0: Motivation
% =============================================================================
\section{Motivation}

% -----------------------------------------------------------------------------
% Slide 3a: Why SSMs? - Problems
% -----------------------------------------------------------------------------
\begin{frame}{Why State Space Models? (1/3)}
    \begin{center}
    \includegraphics[width=0.75\textwidth,height=0.7\textheight,keepaspectratio]{motivation_problems.png}
    \end{center}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 3b: Why SSMs? - Solutions
% -----------------------------------------------------------------------------
\begin{frame}{Why State Space Models? (2/3)}
    \begin{center}
    \includegraphics[width=0.75\textwidth,height=0.7\textheight,keepaspectratio]{motivation_solutions.png}
    \end{center}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 3c: Why SSMs? - Summary
% -----------------------------------------------------------------------------
\begin{frame}{Why State Space Models? (3/3)}
    \textbf{Key Advantages of SSMs:}
    \vspace{0.5cm}
    \begin{itemize}
        \item \textcolor{outputgreen}{\textbf{Efficient Memory}}: Compresses long history into fixed-size state
        \begin{itemize}
            \item Maintains hidden state $h_t$ that encodes full sequence history
            \item Constant memory complexity regardless of sequence length
        \end{itemize}
        \vspace{0.3cm}
        \item \textcolor{outputgreen}{\textbf{Linear Complexity}}: $O(T)$ vs $O(T^2)$ for attention
        \begin{itemize}
            \item Sequential processing enables efficient scanning
            \item Scales to very long sequences (thousands of time steps)
        \end{itemize}
        \vspace{0.3cm}
        \item \textcolor{outputgreen}{\textbf{Long Dependencies}}: Designed for sequences with distant correlations
        \begin{itemize}
            \item HiPPO initialization optimizes for long-range memory
            \item Captures patterns across entire time series
        \end{itemize}
    \end{itemize}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 4a: Running Example - Visualization
% -----------------------------------------------------------------------------
\begin{frame}{Our Running Example (1/2)}
    \begin{center}
    \includegraphics[width=0.85\textwidth,height=0.7\textheight,keepaspectratio]{running_example.png}
    \end{center}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 4b: Running Example - Usage
% -----------------------------------------------------------------------------
\begin{frame}{Our Running Example (2/2)}
    \textbf{Throughout this presentation:}
    \vspace{0.5cm}
    \begin{itemize}
        \item We'll follow a \textbf{univariate time series} of length $T=96$
        \begin{itemize}
            \item Real-world time series with trend, seasonality, and noise
            \item Representative of typical forecasting scenarios
        \end{itemize}
        \vspace{0.4cm}
        \item Focus on the \textcolor{highlight}{\textbf{first 2 values}}: $x_0 = 0.5$ and $x_1 = 0.8$
        \begin{itemize}
            \item These specific values will appear in all numerical examples
            \item Allows tracking data flow through the entire pipeline
        \end{itemize}
        \vspace{0.4cm}
        \item \textbf{Goal}: Understand how SSMs process sequential data
        \begin{itemize}
            \item From raw input to final embedding
            \item Step-by-step mathematical transformations
        \end{itemize}
    \end{itemize}
\end{frame}

% =============================================================================
% Section 1: Architecture
% =============================================================================
\section{Encoder Architecture}

% -----------------------------------------------------------------------------
% Slide 3: High-Level Pipeline
% -----------------------------------------------------------------------------
\begin{frame}{High-Level Encoder Architecture}
    \begin{itemize}
        \item \textbf{Objective}: Map time series $X \in \mathbb{R}^{B \times T \times F}$ to compact embeddings $E \in \mathbb{R}^{B \times D_{emb}}$.
        \item \textbf{Core Components} (from \texttt{mamba\_encoder.yaml}):
        \begin{enumerate}
            \item \textbf{Tokenization}: Slicing time series into windows.
            \item \textbf{Projection}: Mapping tokens to model dimension $D_{model}$.
            \item \textbf{Backbone}: Stack of $L$ Mamba Blocks (SSM).
            \item \textbf{Pooling}: Aggregating sequence info (Mean, Last, or CLS).
        \end{enumerate}
    \end{itemize}

    \vspace{0.5cm}
    \begin{center}
    \begin{tikzpicture}[node distance=1.5cm, auto]
        \node (input) {Input TS};
        \node [draw, right=of input] (tok) {Tokenizer};
        \node [draw, right=of tok] (proj) {Linear/Conv};
        \node [draw, fill=mambapurple!20, right=of proj] (mamba) {Mamba Stack};
        \node [draw, right=of mamba] (pool) {Pooling};
        \node [draw, right=of pool] (out) {Embedding};
        
        \path [->] (input) edge (tok);
        \path [->] (tok) edge (proj);
        \path [->] (proj) edge (mamba);
        \path [->] (mamba) edge (pool);
        \path [->] (pool) edge (out);
    \end{tikzpicture}
    \end{center}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 5a: Encoder Comparison - Architecture
% -----------------------------------------------------------------------------
\begin{frame}{Encoder Variants (1/2)}
    \begin{center}
    \includegraphics[width=0.65\textwidth,height=0.7\textheight,keepaspectratio]{encoder_comparison.png}
    \end{center}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 5b: Encoder Comparison - Details
% -----------------------------------------------------------------------------
\begin{frame}{Encoder Variants (2/2)}
    \begin{columns}[T]
        \begin{column}{0.48\textwidth}
            \textbf{Standard Encoder}
            \vspace{0.3cm}
            \begin{itemize}
                \item \textbf{Input}: Raw sequence values
                \item \textbf{Patching}: Direct 1D windows
                \item \textbf{Projection}: Linear layer (\texttt{nn.Linear})
                \item \textbf{Processing}: Sequential SSM blocks
                \item \textbf{Use Case}: General time series forecasting
            \end{itemize}
        \end{column}
        
        \begin{column}{0.48\textwidth}
            \textbf{Visual Encoder}
            \vspace{0.3cm}
            \begin{itemize}
                \item \textbf{Input}: Time series $\to$ pseudo-images
                \item \textbf{Transform}: Recurrence Plot
                \begin{itemize}
                    \item $R_{ij} = \|x_i - x_j\|$
                \end{itemize}
                \item \textbf{Projection}: 2D convolution (\texttt{Conv2d})
                \item \textbf{Processing}: Vision-style SSM
                \item \textbf{Use Case}: Capturing structural patterns
            \end{itemize}
        \end{column}
    \end{columns}
\end{frame}

% =============================================================================
% Section 2: The Mamba Block
% =============================================================================
\section{The Mamba Block}

% -----------------------------------------------------------------------------
% Slide 5: Block Structure
% -----------------------------------------------------------------------------
\begin{frame}{Inside the Mamba Block}
    The \texttt{MambaBlock} handles the sequence mixing.
    \vspace{0.3cm}
    
    \textbf{Forward Pass} (Simplified):
    \begin{enumerate}
        \item \textbf{Input Projection}: $x \in \mathbb{R}^{B \times T \times D} \to z, x' \in \mathbb{R}^{B \times T \times E}$
        \item \textbf{Convolution}: 1D causal conv on $x'$
        \item \textbf{SSM Processing}: \texttt{\_selective\_scan}$(x', \Delta, A, B, C)$
        \item \textbf{Output Projection}: Combine with gating $z$ and project back
    \end{enumerate}
    
    \vspace{0.3cm}
    \textbf{Selective Parameters}: $\Delta$, $B$, $C$ are \textit{input-dependent}. This is the core innovation!
\end{frame}

% =============================================================================
% Section 3: SSM Mathematics
% =============================================================================
\section{SSM Mathematics}

% -----------------------------------------------------------------------------
% Slide 6: Step 1 - Input
% -----------------------------------------------------------------------------
\begin{frame}{Step 1: Input Sequence}
    \begin{columns}
        \begin{column}{0.5\textwidth}
           The time series $\mathbf{x}$ enters the model. Each time step is a scalar (univariate) or vector (multivariate).
           
           $$ \mathbf{u} : \mathbb{R} \to \mathbb{R}^F $$
           
           For our example: $u_0 = 0.5$, $u_1 = 0.8$, ...
        \end{column}
        \begin{column}{0.5\textwidth}
            \centering
            \includegraphics[width=\textwidth]{step_1_input.png}
        \end{column}
    \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 7: Step 2 - Latent State Evolution
% -----------------------------------------------------------------------------
\begin{frame}{Step 2: Latent State $h(t)$}
    \begin{columns}
        \begin{column}{0.5\textwidth}
           SSM maintains a \textbf{continuous hidden state} that captures history.
           
           $$ \frac{dh}{dt} = \mathbf{A}h(t) + \mathbf{B}u(t) $$
           
           $\mathbf{A}$ encodes state dynamics (HiPPO-initialized). $\mathbf{B}$ maps input influence.
        \end{column}
        \begin{column}{0.5\textwidth}
            \centering
            \includegraphics[width=\textwidth]{step_2_latent.png}
        \end{column}
    \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 8: Discretization Formula
% -----------------------------------------------------------------------------
\begin{frame}[fragile]{Discretization (Zero-Order Hold)}
    To process sampled data with dynamic step sizes $\Delta$, we discretize the continuous system.
    
    Given a step size $\Delta$, the discrete parameters $\overline{\mathbf{A}}$ and $\overline{\mathbf{B}}$ are:
    
    \begin{align}
        \overline{\mathbf{A}} &= \exp(\Delta \mathbf{A}) \\
        \overline{\mathbf{B}} &= (\Delta \mathbf{A})^{-1} (\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B} \nonumber \\
                              &= \mathbf{A}^{-1} (\overline{\mathbf{A}} - \mathbf{I}) \cdot \mathbf{B}
    \end{align}
    
    This matches the specific implementation in \texttt{mamba\_block.py}:
\begin{verbatim}
integral = torch.linalg.solve(A_expand, A_expm - eye)
B_disc = torch.bmm(integral, B_expand)
\end{verbatim}
\end{frame}

% -----------------------------------------------------------------------------
% Slide: Section Summary - SSM Math (1/2)
% -----------------------------------------------------------------------------
\begin{frame}{Section Summary: SSM Mathematics (1/2)}
    \begin{block}{Continuous System}
        $\frac{dh}{dt} = \mathbf{A}h + \mathbf{B}u$ \ (Differential equation)
        \vspace{0.2cm}
        \\ Models smooth evolution of hidden state over continuous time
    \end{block}
    \vspace{0.3cm}
    
    \begin{block}{Discretization (Zero-Order Hold)}
        $\overline{\mathbf{A}} = \exp(\Delta \mathbf{A})$, \quad
        $\overline{\mathbf{B}} = \mathbf{A}^{-1}(\overline{\mathbf{A}} - \mathbf{I})\mathbf{B}$
        \vspace{0.2cm}
        \\ Converts continuous dynamics to discrete time steps of size $\Delta$
    \end{block}
    \vspace{0.3cm}
    
    \begin{block}{Discrete Recurrence}
        $h_t = \overline{\mathbf{A}}h_{t-1} + \overline{\mathbf{B}}u_t$ \ (State update)
        \\ $y_t = \mathbf{C}h_t$ \ (Output projection)
        \vspace{0.2cm}
        \\ Enables efficient sequential processing of sampled data
    \end{block}
\end{frame}

% -----------------------------------------------------------------------------
% Slide: Section Summary - SSM Math (2/2)
% -----------------------------------------------------------------------------
\begin{frame}{Section Summary: SSM Mathematics (2/2)}
    \begin{center}
    \includegraphics[width=0.75\textwidth,height=0.65\textheight,keepaspectratio]{continuous_vs_discrete.png}
    \end{center}
    
    \vspace{0.2cm}
    \textbf{Key Concept}: Zero-Order Hold (ZOH) maintains input constant between samples
\end{frame}

% =============================================================================
% Section 4: Data Processing Pipeline
% =============================================================================
\section{Data Processing (Recurrence Plots)}

% -----------------------------------------------------------------------------
% Slide 9: Patching
% -----------------------------------------------------------------------------
\begin{frame}{Step 3: Time Series Patching}
    \begin{columns}
        \begin{column}{0.4\textwidth}
           The raw time series is typically long and noisy. 
           
           We extract \textbf{overlapping patches} of fixed length (e.g., 96 steps).
           
           This is analogous to image patching in Vision Transformers.
        \end{column}
        \begin{column}{0.6\textwidth}
            \centering
            \includegraphics[width=\textwidth]{step_3_patching.png}
        \end{column}
    \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 10: Feature Extraction
% -----------------------------------------------------------------------------
\begin{frame}{Step 4: Feature Extraction}
    \begin{columns}
        \begin{column}{0.5\textwidth}
           From each patch, we can extract:
           \begin{itemize}
               \item \textbf{Raw values}: Direct encoding
               \item \textbf{Statistical features}: Mean, variance, etc.
               \item \textbf{Structural features}: Recurrence structure
           \end{itemize}
           
           The Visual Encoder uses the last option.
        \end{column}
        \begin{column}{0.5\textwidth}
            \centering
            \includegraphics[width=\textwidth]{step_4_extraction.png}
        \end{column}
    \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 11: Recurrence Plot Generation
% -----------------------------------------------------------------------------
\begin{frame}{Step 5: Visual Transformation (RP)}
    \begin{columns}
        \begin{column}{0.5\textwidth}
           We convert the 1D patch into a 2D \textbf{Recurrence Plot}.
           
           $$ R_{i,j} = \| \mathbf{x}_i - \mathbf{x}_j \| $$
           
           \textbf{Numerical Example:}
           Sequence patch (first 2 values): $\mathbf{x} = [0.5, 0.8]$
           $$ \mathbf{R} = \begin{pmatrix} |0.5-0.5| & |0.5-0.8| \\ |0.8-0.5| & |0.8-0.8| \end{pmatrix} $$
           $$ \mathbf{R} = \begin{pmatrix} 0 & 0.3 \\ 0.3 & 0 \end{pmatrix} $$
        \end{column}
        \begin{column}{0.5\textwidth}
            \centering
            \includegraphics[width=\textwidth]{step_5_rp.png}
        \end{column}
    \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 12: The Selective Scan Algorithm
% -----------------------------------------------------------------------------
\section{Algorithm}
\begin{frame}{Algorithm: Selective Scan}
    \textbf{Logic verified in:} \texttt{src/models/mamba\_block.py}
    
    \begin{algorithm}[H]
    \caption{Selective Scan with HiPPO-Initialized SSM}
    \label{alg:selective-scan}
    \begin{algorithmic}[1]
    \footnotesize
    \REQUIRE Input $\mathbf{x} \in \mathbb{R}^{B \times T \times D}$, step sizes $\boldsymbol{\delta} \in \mathbb{R}^{B \times T \times 1}$
    \ENSURE Output $\mathbf{y} \in \mathbb{R}^{B \times T \times D}$
    \STATE Precompute $\mathbf{A}, \mathbf{B}, \mathbf{C}$ (HiPPO)
    \FOR{$t = 0$ \textbf{to} $T-1$}
        \STATE $\mathbf{u}_t \gets \mathbf{x}[:, t, :]$; $\Delta_t \gets \boldsymbol{\delta}[:, t, :]$ 
        \STATE $\widetilde{\mathbf{A}} \gets \Delta_t \cdot \mathbf{A}$ \COMMENT{Scale A by time step}
        \STATE $\mathbf{A}_d \gets \exp(\widetilde{\mathbf{A}})$ \COMMENT{Discretize A}
        \STATE $\mathbf{h} \gets \mathbf{A}_d\, \mathbf{h} + \mathbf{B}_d\, \mathbf{u}_t$ \COMMENT{Update State}
        \STATE $\mathbf{y}_t \gets \mathbf{C}\, \mathbf{h}$ \COMMENT{Project to Output}
    \ENDFOR
    \end{algorithmic}
    \end{algorithm}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 13a: Numerical Trace - Visualization
% -----------------------------------------------------------------------------
\begin{frame}{Numerical Walkthrough: Selective Scan (1/2)}
    \begin{center}
    \includegraphics[width=0.85\textwidth,height=0.75\textheight,keepaspectratio]{algorithm_trace_extended.png}
    \end{center}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 13b: Numerical Trace - Analysis
% -----------------------------------------------------------------------------
\begin{frame}{Numerical Walkthrough: Selective Scan (2/2)}
    \textbf{Key Insight:} At $t=1$, state $h_1$ retains memory from both $u_0$ and $u_1$!
    \vspace{0.5cm}
    
    \begin{itemize}
        \item \textcolor{inputblue}{\textbf{Input Sequence}}:
        \begin{itemize}
            \item $u_0 = 0.5$, $u_1 = 0.8$ (from our running example)
            \item Step size: $\Delta = 0.5$
        \end{itemize}
        \vspace{0.3cm}
        
        \item \textcolor{statepurple}{\textbf{State Evolution}}:
        \begin{itemize}
            \item $h_0 = \overline{\mathbf{B}}u_0 \approx [0.195, 0.145]^T$
            \item $h_1 = \overline{\mathbf{A}}h_0 + \overline{\mathbf{B}}u_1 \approx [0.431, 0.318]^T$
            \item State accumulates information: $h_1$ depends on both $u_0$ and $u_1$
        \end{itemize}
        \vspace{0.3cm}
        
        \item \textcolor{outputgreen}{\textbf{Output Sequence}}:
        \begin{itemize}
            \item $y_0 = \mathbf{C}h_0 = 0.195$
            \item $y_1 = \mathbf{C}h_1 = 0.431$
        \end{itemize}
    \end{itemize}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 14: Implementation Details
% -----------------------------------------------------------------------------
\begin{frame}{Implementation Specifics}
    Based on \texttt{mamba\_block.py}:
    \begin{itemize}
        \item \textbf{HiPPO Initialization}: Matrix $\mathbf{A}$ is initialized using the Legendre-S (LegS) measure to handle long-term dependencies.
        \item \textbf{Parallelism vs. Scanning}:
        \begin{itemize}
            \item The Python implementation performs a \textbf{sequential loop} (Line 144 in \texttt{mamba\_block.py}).
            \item Optimized CUDA implementations (not currently used) usually perform a parallel associative scan.
        \end{itemize}
        \item \textbf{Numerical Stability}:
        \begin{itemize}
            \item $\Delta$ is clamped: $\Delta \in [10^{-4}, 3.0]$.
            \item \texttt{torch.linalg.solve} used instead of explicit inverse for $\mathbf{A}^{-1}$ to ensure stability.
        \end{itemize}
    \end{itemize}
\end{frame}

% -----------------------------------------------------------------------------
% Slide: Final Summary
% -----------------------------------------------------------------------------
\begin{frame}{Summary: State Space Models for Time Series}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \textbf{Core Concepts:}
            \begin{enumerate}
                \item \textcolor{highlight}{\textbf{Motivation}}: Efficient long-range dependencies
                \item \textcolor{inputblue}{\textbf{SSM Math}}: Continuous $\to$ Discrete via ZOH
                \item \textcolor{patchorange}{\textbf{Selective Scan}}: Dynamic state updates
                \item \textcolor{statepurple}{\textbf{Visual Encoding}}: Recurrence Plots for 2D patterns
            \end{enumerate}
        \end{column}
        \begin{column}{0.5\textwidth}
            \textbf{Implementation:}
            \begin{itemize}
                \item HiPPO initialization for $\mathbf{A}$
                \item Parallel training + Sequential inference
                \item Two encoder variants: Standard \& Visual
                \item Numerically stable via \texttt{torch.linalg.solve}
            \end{itemize}
        \end{column}
    \end{columns}
    
    \vspace{0.5cm}
    \begin{center}
        \textbf{Running Example:} Time series ($T=96$) with $x_0=0.5, x_1=0.8$
        \\ $\Rightarrow$ State memory: $h_1$ encodes both past and present!
    \end{center}
\end{frame}

\begin{frame}
    \centering \Huge Questions?
\end{frame}

\end{document}
"""


def create_directory_structure():
    """Create the output directory structure."""
    print(f"Creating directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("✓ Directory created")


def generate_visualizations():
    """Generate all required visualization images."""
    print("\nGenerating visualizations...")
    
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11
    plt.rcParams['figure.dpi'] = 150
    
    # Consistent color scheme
    COLORS = {
        'input': '#2E86AB',
        'state': '#A23B72',
        'output': '#06A77D',
        'patch': '#F18F01',
        'highlight': '#C73E1D',
        'background': '#F6F6F6'
    }
    
    # Generate time series
    np.random.seed(42)
    t = np.linspace(0, 10, 96)
    trend = 0.05 * t
    seasonality = 0.3 * np.sin(2 * np.pi * t / 2) + 0.2 * np.sin(2 * np.pi * t / 0.5)
    noise = 0.1 * np.random.randn(96)
    time_series = 0.5 + trend + seasonality + noise
    time_series[0] = 0.5
    time_series[1] = 0.8
    
    # 1. Running Example
    print("  - running_example.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(96), time_series, color=COLORS['input'], linewidth=2, alpha=0.7, label='Time Series (96 values)')
    ax.scatter([0, 1], [time_series[0], time_series[1]], color=COLORS['highlight'], s=200, zorder=5, 
               label='First 2 values', edgecolors='black', linewidths=2)
    ax.annotate(f'x_0 = {time_series[0]:.1f}', xy=(0, time_series[0]), xytext=(5, time_series[0] + 0.3),
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['highlight'], alpha=0.7),
                arrowprops=dict(arrowstyle='->', color=COLORS['highlight'], lw=2))
    ax.annotate(f'x_1 = {time_series[1]:.1f}', xy=(1, time_series[1]), xytext=(6, time_series[1] + 0.3),
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['highlight'], alpha=0.7),
                arrowprops=dict(arrowstyle='->', color=COLORS['highlight'], lw=2))
    ax.set_xlabel('Time Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Value', fontsize=13, fontweight='bold')
    ax.set_title('Running Example: Univariate Time Series (Length=96)', fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 96)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'running_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Continuous vs Discrete
    print("  - continuous_vs_discrete.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    t_cont = np.linspace(0, 3, 300)
    x_cont = np.exp(-0.5 * t_cont) * np.cos(2 * np.pi * t_cont)
    ax1.plot(t_cont, x_cont, color=COLORS['state'], linewidth=2.5)
    ax1.set_title('Continuous Time System', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time t (continuous)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('State h(t)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.5, 0.7, 'dh/dt = Ah + Bu', fontsize=13, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    t_discrete = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    x_discrete = np.exp(-0.5 * t_discrete) * np.cos(2 * np.pi * t_discrete)
    for i in range(len(t_discrete) - 1):
        ax2.hlines(x_discrete[i], t_discrete[i], t_discrete[i+1], 
                  colors=COLORS['input'], linewidth=3, label='ZOH' if i == 0 else '')
        ax2.plot([t_discrete[i+1]], [x_discrete[i]], 'o', color=COLORS['input'], markersize=4)
    ax2.plot(t_discrete, x_discrete, 'o', color=COLORS['highlight'], markersize=10, 
            label='Sampled States', zorder=5, markeredgecolor='black', markeredgewidth=1.5)
    ax2.annotate('', xy=(1.0, -0.9), xytext=(0.5, -0.9),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax2.text(0.75, -1.0, 'Delta', fontsize=13, ha='center', fontweight='bold')
    ax2.set_title('Discrete Time System (ZOH)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time t (discrete)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('State h_t', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.2, 0.7, 'h_t = A_bar * h_{t-1} + B_bar * u_t', fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'continuous_vs_discrete.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Algorithm Trace Extended
    print("  - algorithm_trace_extended.png")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Selective Scan Algorithm: Step-by-Step Trace', fontsize=16, fontweight='bold', y=0.98)
    
    A_disc0 = np.array([[0.61, 0], [0.30, 0.61]])
    B_disc0 = np.array([[0.39], [0.29]])
    C = np.array([[1, 0]])
    u0, u1 = 0.5, 0.8
    delta0 = 0.5
    h0 = B_disc0 * u0
    y0 = C @ h0
    h1 = A_disc0 @ h0 + B_disc0 * u1
    y1 = C @ h1
    
    # t=0 plots
    ax = axes[0, 0]
    ax.bar(['u_0', 'Delta_0'], [u0, delta0], color=[COLORS['input'], COLORS['patch']], 
          edgecolor='black', linewidth=2)
    ax.set_title('t=0: Input', fontweight='bold', fontsize=12)
    ax.set_ylabel('Value', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[0, 1]
    im = ax.imshow(np.column_stack([A_disc0, B_disc0]), cmap='Purples', vmin=0, vmax=0.8, aspect='auto')
    ax.set_title('t=0: Discretized [A_bar | B_bar]', fontweight='bold', fontsize=12)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['A[:,0]', 'A[:,1]', 'B'])
    for i in range(2):
        for j in range(3):
            val = np.column_stack([A_disc0, B_disc0])[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontweight='bold', fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[0, 2]
    ax.barh(['h_0[0]', 'h_0[1]'], h0.flatten(), color=COLORS['state'], edgecolor='black', linewidth=2)
    ax.set_title('t=0: State h_0', fontweight='bold', fontsize=12)
    ax.set_xlabel('Value', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.text(0.15, 1.5, f'y_0 = {y0[0, 0]:.3f}', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=COLORS['output'], alpha=0.7))
    
    # t=1 plots
    ax = axes[1, 0]
    ax.bar(['u_1', 'Delta_1'], [u1, delta0], color=[COLORS['input'], COLORS['patch']], 
          edgecolor='black', linewidth=2)
    ax.set_title('t=1: Input', fontweight='bold', fontsize=12)
    ax.set_ylabel('Value', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[1, 1]
    ax.axis('off')
    ax.text(0.5, 0.7, 'State Update:', ha='center', fontsize=13, fontweight='bold')
    ax.text(0.5, 0.5, 'h_1 = A_bar * h_0 + B_bar * u_1', ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=2))
    ax.text(0.5, 0.25, 'Memory of h_0 retained!', ha='center', fontsize=11, 
           style='italic', color=COLORS['highlight'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    ax = axes[1, 2]
    x_pos = np.arange(2)
    width = 0.35
    ax.barh(x_pos - width/2, h0.flatten(), width, label='h_0 (prev)', 
           color=COLORS['state'], alpha=0.5, edgecolor='black', linewidth=1.5)
    ax.barh(x_pos + width/2, h1.flatten(), width, label='h_1 (curr)', 
           color=COLORS['state'], edgecolor='black', linewidth=2)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(['h[0]', 'h[1]'])
    ax.set_title('t=1: State Evolution', fontweight='bold', fontsize=12)
    ax.set_xlabel('Value', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.text(0.4, 1.7, f'y_1 = {y1[0, 0]:.3f}', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=COLORS['output'], alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'algorithm_trace_extended.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Encoder Comparison - Fixed arrows
    print("  - encoder_comparison.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Encoder Architectures Comparison', fontsize=16, fontweight='bold')
    
    for ax, title, boxes in [(ax1, 'Standard Encoder', [
        (1, 8.5, 8, 1.2, 'Input Time Series\n[B x T x F]', COLORS['input']),
        (1, 6.5, 8, 1.2, 'Patching\n(1D Windows)', COLORS['patch']),
        (1, 4.5, 8, 1.2, 'Linear Projection\nR^F -> R^D', 'lightgray'),
        (1, 2.5, 8, 1.2, 'Mamba SSM Stack\n(Selective Scan)', COLORS['state']),
        (1, 0.5, 8, 1.2, 'Pooling -> Embedding', COLORS['output']),
    ]), (ax2, 'Visual Encoder (RP)', [
        (1, 8.5, 8, 1.2, 'Input Time Series\n[B x T x F]', COLORS['input']),
        (1, 6.5, 8, 1.2, 'Patching\n(1D Windows)', COLORS['patch']),
        (1, 4.5, 8, 1.2, 'Recurrence Plot\nR_ij = ||x_i - x_j|| (2D)', COLORS['highlight']),
        (1, 2.5, 8, 1.2, 'Mamba SSM Stack\n(2D Vision-style)', COLORS['state']),
        (1, 0.5, 8, 1.2, 'Pooling -> Embedding', COLORS['output']),
    ])]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 11)
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        for i, (x, y, w, h, text, color) in enumerate(boxes):
            ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2.5, alpha=0.7))
            ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')
            # Draw arrow between boxes (not from box edge)
            if i < len(boxes) - 1:
                # Arrow starts below current box and points to top of next box
                arrow_x = x + w/2
                arrow_y_start = y  # Bottom of current box
                arrow_y_end = boxes[i+1][1] + boxes[i+1][3]  # Top of next box
                ax.annotate('', xy=(arrow_x, arrow_y_end), xytext=(arrow_x, arrow_y_start),
                           arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'encoder_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5a. Motivation Problems
    print("  - motivation_problems.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Challenges in Time Series Modeling', fontsize=16, fontweight='bold')
    
    ax1.set_title('Problem: Long-Range Dependencies', fontsize=13, fontweight='bold')
    np.random.seed(42)
    t_prob = np.arange(100)
    signal = np.sin(0.1 * t_prob) + 0.3 * np.random.randn(100)
    ax1.plot(t_prob, signal, color=COLORS['input'], linewidth=2)
    ax1.axvspan(0, 20, alpha=0.3, color=COLORS['highlight'], label='Local context')
    ax1.axvspan(80, 100, alpha=0.3, color=COLORS['output'], label='Prediction target')
    ax1.arrow(10, 1.5, 65, 0, head_width=0.2, head_length=3, fc='red', ec='red', linewidth=3)
    ax1.text(42, 1.8, 'Long dependency', ha='center', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower left')
    ax1.set_xlabel('Time Step', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Value', fontweight='bold', fontsize=11)
    ax1.grid(alpha=0.3)
    
    ax2.set_title('Problem: Attention Complexity', fontsize=13, fontweight='bold')
    sequence_lengths = [64, 128, 256, 512, 1024, 2048]
    attention_ops = [n**2 for n in sequence_lengths]
    linear_ops = [n for n in sequence_lengths]
    ax2.plot(sequence_lengths, attention_ops, 'o-', linewidth=3, label='Attention: O(T²)', 
            color='red', markersize=10)
    ax2.plot(sequence_lengths, linear_ops, 's-', linewidth=3, label='SSM: O(T)', 
            color=COLORS['output'], markersize=10)
    ax2.set_xlabel('Sequence Length T', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Operations (normalized)', fontweight='bold', fontsize=11)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.set_yscale('log')
    ax2.grid(alpha=0.3)
    ax2.text(0.5, 0.95, 'Quadratic explosion vs Linear scaling', transform=ax2.transAxes,
            ha='center', va='top', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'motivation_problems.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5b. Motivation Solutions
    print("  - motivation_solutions.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('State Space Model Solutions', fontsize=16, fontweight='bold')
    
    ax1.set_title('Solution: Continuous State Memory', fontsize=13, fontweight='bold')
    ax1.axis('off')
    ax1.text(0.5, 0.8, 'SSM maintains hidden state h_t', ha='center', fontsize=12, fontweight='bold')
    ax1.text(0.5, 0.6, 'h_t = f(A, B, h_{t-1}, x_t)', ha='center', fontsize=14,
            bbox=dict(boxstyle='round', facecolor=COLORS['state'], alpha=0.4, edgecolor='black', linewidth=2))
    ax1.text(0.5, 0.35, 'Benefits:', ha='center', fontsize=11, fontweight='bold')
    ax1.text(0.5, 0.2, '• Compresses full history efficiently\n• Constant memory complexity\n• Captures long-range patterns', 
            ha='center', fontsize=10, style='italic')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    ax2.set_title('Solution: Efficient Sequential Scan', fontsize=13, fontweight='bold')
    ax2.axis('off')
    ax2.text(0.5, 0.8, 'Sequential processing enables:', ha='center', fontsize=12, fontweight='bold')
    ax2.text(0.5, 0.6, 'Linear time: O(T)\nParallel training\nHardware-optimized', 
            ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=2))
    ax2.text(0.5, 0.35, 'Result:', ha='center', fontsize=11, fontweight='bold')
    ax2.text(0.5, 0.2, 'Scales to very long sequences!\n(thousands of time steps)', ha='center', fontsize=11, 
            style='italic', color=COLORS['output'], fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'motivation_solutions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Step images with actual visualizations
    print("  - step_1_input.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(10), time_series[:10], 'o-', color=COLORS['input'], linewidth=3, markersize=10)
    ax.scatter([0, 1], [time_series[0], time_series[1]], color=COLORS['highlight'], s=300, zorder=5,
               edgecolors='black', linewidths=2.5)
    ax.set_title('Step 1: Input Sequence u(t)', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Time Step t', fontsize=13, fontweight='bold')
    ax.set_ylabel('Input Value u_t', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.text(0, time_series[0] + 0.25, f'u_0 = {time_series[0]:.1f}', fontsize=12, fontweight='bold',
           ha='center', bbox=dict(boxstyle='round', facecolor=COLORS['highlight'], alpha=0.7))
    ax.text(1, time_series[1] + 0.25, f'u_1 = {time_series[1]:.1f}', fontsize=12, fontweight='bold',
           ha='center', bbox=dict(boxstyle='round', facecolor=COLORS['highlight'], alpha=0.7))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'step_1_input.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  - step_2_latent.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    t_cont = np.linspace(0, 5, 200)
    h1 = 0.7 * np.exp(-0.3 * t_cont) * np.cos(2 * np.pi * 0.5 * t_cont)
    h2 = 0.5 * np.exp(-0.2 * t_cont) * np.sin(2 * np.pi * 0.7 * t_cont)
    ax.plot(t_cont, h1, label='h[0] (dimension 1)', color=COLORS['state'], linewidth=2.5)
    ax.plot(t_cont, h2, label='h[1] (dimension 2)', color=COLORS['state'], linewidth=2.5, linestyle='--')
    ax.fill_between(t_cont, h1, alpha=0.2, color=COLORS['state'])
    ax.set_title('Step 2: Latent State Evolution h(t)', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Time t', fontsize=13, fontweight='bold')
    ax.set_ylabel('Hidden State h(t)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.text(2.5, 0.6, 'dh/dt = Ah(t) + Bu(t)', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'step_2_latent.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  - step_3_patching.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(96), time_series, color='gray', linewidth=1, alpha=0.5, label='Full series')
    patch_start = 0
    patch_size = 16
    for i in range(3):
        start = i * 12
        end = start + patch_size
        ax.axvspan(start, end, alpha=0.3, color=[COLORS['patch'], COLORS['input'], COLORS['output']][i])
        ax.text(start + patch_size/2, 1.5, f'Patch {i+1}', ha='center', fontsize=11, fontweight='bold')
    ax.set_title('Step 3: Time Series Patching', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Time Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Value', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'step_3_patching.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  - step_4_extraction.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Step 4: Feature Extraction from Patch', fontsize=15, fontweight='bold')
    patch_data = time_series[:16]
    ax1.plot(range(16), patch_data, 'o-', color=COLORS['patch'], linewidth=2, markersize=8)
    ax1.set_title('Original Patch', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Position', fontweight='bold')
    ax1.set_ylabel('Value', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    features = ['Mean', 'Std', 'Min', 'Max', 'Trend']
    values = [np.mean(patch_data), np.std(patch_data), np.min(patch_data), 
             np.max(patch_data), (patch_data[-1] - patch_data[0])]
    ax2.barh(features, values, color=[COLORS['output'], COLORS['state'], COLORS['highlight'], 
                                      COLORS['input'], COLORS['patch']], edgecolor='black', linewidth=2)
    ax2.set_title('Extracted Features', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Value', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'step_4_extraction.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  - step_5_rp.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Step 5: Recurrence Plot Generation', fontsize=15, fontweight='bold')
    
    patch_small = time_series[:8]
    ax1.plot(range(8), patch_small, 'o-', color=COLORS['patch'], linewidth=2.5, markersize=10)
    ax1.set_title('1D Time Series Patch', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Position', fontweight='bold')
    ax1.set_ylabel('Value', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    rp_matrix = np.abs(patch_small[:, None] - patch_small[None, :])
    im = ax2.imshow(rp_matrix, cmap='Blues', origin='lower')
    ax2.set_title('2D Recurrence Plot: R_ij = |x_i - x_j|', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Position j', fontweight='bold')
    ax2.set_ylabel('Position i', fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Distance')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'step_5_rp.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ All visualizations generated")


def write_latex_file():
    """Write the LaTeX source file."""
    print("\nWriting LaTeX file...")
    latex_path = OUTPUT_DIR / LATEX_FILE
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(LATEX_CONTENT)
    print(f"✓ LaTeX file written: {latex_path}")


def compile_pdf():
    """Compile the LaTeX file to PDF using pdflatex."""
    print("\nCompiling PDF...")
    latex_path = OUTPUT_DIR / LATEX_FILE
    
    # Run pdflatex twice for cross-references
    for i in range(2):
        print(f"  Pass {i+1}/2...")
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', LATEX_FILE],
            cwd=OUTPUT_DIR,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"✗ Compilation failed (pass {i+1})")
            print("Error output:")
            print(result.stdout[-1000:])  # Last 1000 chars
            return False
    
    pdf_path = OUTPUT_DIR / "ssm_presentation.pdf"
    if pdf_path.exists():
        file_size = pdf_path.stat().st_size / 1024  # KB
        print(f"✓ PDF compiled successfully: {pdf_path} ({file_size:.1f} KB)")
        return True
    else:
        print("✗ PDF file not found after compilation")
        return False


def clean_auxiliary_files():
    """Remove auxiliary LaTeX files."""
    print("\nCleaning auxiliary files...")
    extensions = ['.aux', '.log', '.nav', '.out', '.snm', '.toc', '.vrb']
    for ext in extensions:
        file_path = OUTPUT_DIR / f"ssm_presentation{ext}"
        if file_path.exists():
            file_path.unlink()
            print(f"  Removed: {file_path.name}")
    print("✓ Cleanup complete")


def main():
    """Main execution function."""
    print("=" * 60)
    print("SSM Presentation Builder")
    print("=" * 60)
    
    try:
        # Step 1: Create directory structure
        create_directory_structure()
        
        # Step 2: Generate visualizations
        generate_visualizations()
        
        # Step 3: Write LaTeX file
        write_latex_file()
        
        # Step 4: Compile PDF
        success = compile_pdf()
        
        # Step 5: Clean up
        clean_auxiliary_files()
        
        print("\n" + "=" * 60)
        if success:
            print("✓ BUILD SUCCESSFUL")
            print(f"Output: {OUTPUT_DIR / 'ssm_presentation.pdf'}")
        else:
            print("✗ BUILD FAILED")
            print("Check error messages above")
        print("=" * 60)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
