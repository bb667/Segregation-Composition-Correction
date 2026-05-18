# =============================================================================
# compositional_correction.R
# =============================================================================
#
# Two-group segregation indices with the composition-invariance correction
# from Barron et al. (2026), "When All Measures Fail the Same Way".
#
# Quick start
# -----------
#   source("compositional_correction.R")
#
#   # Three accepted input forms -- all equivalent:
#   dissimilarity(df, "white", "hispanic")   # data.frame + column names
#   dissimilarity(M)                          # matrix of shape (n, 2)
#   dissimilarity(vec_a, vec_b)               # two numeric vectors
#
#   # Pick an index by short code or by name (case-insensitive):
#   segregation_index(df, "white", "hispanic", index = "D")
#   segregation_index(df, "white", "hispanic", index = "dissimilarity")
#
#   # Compositional correction:
#   dissimilarity(df, "white", "hispanic", target = 0.50)    # one value
#   composition_curve(df, "white", "hispanic", index = "D")  # whole curve
#
#   # Plot the curve. By default the curve runs from the 10th to the 90th
#   # population-weighted percentile of composition -- i.e. across the
#   # central 80% of the population, where the data is dense enough to
#   # support extrapolation. Override with `population_percentile =`, or
#   # pass NULL to plot the full sweep:
#   plot_composition_curve(df, "white", "hispanic", index = "D")
#
# Available indices
# -----------------
#   'D'   or 'dissimilarity'
#   'S'   or 'separation'            (variance ratio)
#   'H'   or 'entropy'               (Theil H)
#   'R'   or 'hutchens'              (square-root index)
#   'G'   or 'gini'
#   'Iso' or 'isolation'             (xPx for the focal group)
#
# Public functions
# ----------------
#   segregation_index(), dissimilarity(), separation(), entropy_index(),
#   hutchens_r(), gini_index(), isolation(), composition_curve(),
#   plot_composition_curve()
#
# Reference
# ---------
# Barron, B., Hall, M., Rich, P., Cohen, I., Arias, T. A. (2026).
# When All Measures Fail the Same Way: A Correction to Segregation Trends.
# =============================================================================


# -----------------------------------------------------------------------------
# Index name resolution: short codes and long names, case-insensitive.
# -----------------------------------------------------------------------------

.INDEX_ALIASES <- c(
  "d" = "D",   "dissimilarity" = "D",
  "s" = "S",   "separation" = "S",   "variance_ratio" = "S",
  "h" = "H",   "entropy" = "H",      "theil" = "H",
  "r" = "R",   "hutchens" = "R",     "hutchens_r" = "R", "square_root" = "R",
  "g" = "G",   "gini" = "G",
  "iso" = "Iso", "isolation" = "Iso", "xpx" = "Iso"
)

.INDEX_LONG_NAMES <- c(
  "D"   = "Dissimilarity",
  "S"   = "Separation (variance ratio)",
  "H"   = "Entropy (Theil H)",
  "R"   = "Hutchens R (square root)",
  "G"   = "Gini",
  "Iso" = "Isolation (xPx)"
)

.canonical_index <- function(name) {
  if (!is.character(name) || length(name) != 1L) {
    stop("Index name must be a single string.", call. = FALSE)
  }
  key <- tolower(gsub("[-[:space:]]+", "_", trimws(name)))
  if (!key %in% names(.INDEX_ALIASES)) {
    codes <- sort(unique(unname(.INDEX_ALIASES)))
    stop(sprintf(
      "Unknown index '%s'. Valid short codes: %s. Long names also work (e.g. 'dissimilarity', 'gini', 'isolation').",
      name, paste(sprintf("'%s'", codes), collapse = ", ")
    ), call. = FALSE)
  }
  unname(.INDEX_ALIASES[key])
}


# -----------------------------------------------------------------------------
# Input handling: accept any of three data forms.
# -----------------------------------------------------------------------------

# Convert any accepted input form into pi (focal share per neighborhood) and
# t (pairwise total). Empty neighborhoods (a + b == 0) are dropped.
#
# Form 1 -- data.frame + two column names:
#     .to_pi_t(df, "col_a", "col_b")
#
# Form 2 -- 2-column matrix, column 1 = group A, column 2 = group B:
#     .to_pi_t(M)
#
# Form 3 -- two numeric vectors of equal length, focal first:
#     .to_pi_t(vec_a, vec_b)
.to_pi_t <- function(data, a = NULL, b = NULL) {
  if (is.data.frame(data)) {
    if (is.null(a) || is.null(b)) {
      stop("data.frame input requires two column names: f(df, 'col_a', 'col_b').",
           call. = FALSE)
    }
    a_vec <- as.numeric(data[[a]])
    b_vec <- as.numeric(data[[b]])
  } else if (is.matrix(data)) {
    if (!is.null(a) || !is.null(b)) {
      stop("Matrix input takes no extra arguments; column 1 is group A and column 2 is group B.",
           call. = FALSE)
    }
    if (ncol(data) != 2L) {
      stop(sprintf("Matrix input must have 2 columns; got %d.", ncol(data)),
           call. = FALSE)
    }
    a_vec <- as.numeric(data[, 1])
    b_vec <- as.numeric(data[, 2])
  } else if (is.numeric(data) && is.null(dim(data))) {
    if (is.null(a)) {
      stop("Vector input requires a second vector for group B: f(vec_a, vec_b).",
           call. = FALSE)
    }
    if (!is.null(b)) {
      stop("Vector input takes exactly one additional argument (the second vector).",
           call. = FALSE)
    }
    a_vec <- as.numeric(data)
    b_vec <- as.numeric(a)
    if (length(a_vec) != length(b_vec)) {
      stop(sprintf("Group vectors must be the same length; got %d and %d.",
                   length(a_vec), length(b_vec)), call. = FALSE)
    }
  } else {
    stop("Unsupported input type. Expected data.frame, matrix, or numeric vector.",
         call. = FALSE)
  }

  if (any(a_vec < 0, na.rm = TRUE) || any(b_vec < 0, na.rm = TRUE)) {
    stop("Population counts must be non-negative.", call. = FALSE)
  }

  t_vec <- a_vec + b_vec
  keep <- !is.na(t_vec) & t_vec > 0
  if (!any(keep)) {
    stop("No neighborhoods with non-zero total of the two groups.", call. = FALSE)
  }

  list(pi = a_vec[keep] / t_vec[keep], t = t_vec[keep])
}


# -----------------------------------------------------------------------------
# Core math: tilting, root finding, and the index formulas.
# -----------------------------------------------------------------------------

# exp(v * pi), shifted for numerical stability. Overall scale cancels wherever
# weights appear.
.weights <- function(pi, v) {
  log_w <- v * pi
  log_w <- log_w - max(log_w)
  exp(log_w)
}

# Population-weighted mean of pi after tilting weights by exp(v * pi).
.tilted_mean <- function(pi, t, v) {
  w <- .weights(pi, v)
  sum(t * w * pi) / sum(t * w)
}

# Find the tilting parameter v whose induced composition equals target. The
# map v -> tilted_mean(v) is strictly monotone, so uniroot is well-posed.
.solve_v <- function(pi, t, target, v_bracket = c(-50, 50)) {
  lo <- min(pi); hi <- max(pi)
  if (!(lo < target && target < hi)) {
    stop(sprintf(
      "Target composition %.4f is outside the achievable range [%.4f, %.4f] of observed neighborhood shares. The correction can only redistribute mass within the support of pi.",
      target, lo, hi), call. = FALSE)
  }
  f <- function(v) .tilted_mean(pi, t, v) - target
  # Tight tolerance to match scipy.brentq's default precision.
  uniroot(f, interval = v_bracket, tol = 1e-12)$root
}

# Evaluate one segregation index from (pi, t) at composition mu.
#
# For the standard (unadjusted) value, pass raw t and mu = weighted.mean(pi, t).
# For the corrected value, pass t * exp(v*pi) and mu = target. Everything else
# is the same textbook formula.
.compute <- function(pi, t, mu, code) {
  if (mu <= 0 || mu >= 1) return(0)
  T_ <- sum(t)

  if (code == "D") {
    return(sum(t * abs(pi - mu)) / (2 * T_ * mu * (1 - mu)))
  }

  if (code == "S") {
    return(sum(t * (pi - mu)^2) / (T_ * mu * (1 - mu)))
  }

  if (code == "H") {
    E <- -mu * log(mu) - (1 - mu) * log(1 - mu)
    if (E <= 0) return(0)
    term_a <- numeric(length(pi))
    mask_a <- pi > 0
    term_a[mask_a] <- pi[mask_a] * log(pi[mask_a] / mu)
    term_b <- numeric(length(pi))
    mask_b <- pi < 1
    term_b[mask_b] <- (1 - pi[mask_b]) * log((1 - pi[mask_b]) / (1 - mu))
    return(sum(t * (term_a + term_b)) / (T_ * E))
  }

  if (code == "R") {
    return(1 - sum(t * sqrt(pi * (1 - pi))) / (T_ * sqrt(mu * (1 - mu))))
  }

  if (code == "G") {
    # O(n log n) sorted-cumsum identity.
    ord  <- order(pi)
    pi_s <- pi[ord]
    t_s  <- t[ord]
    cum_t   <- c(0, cumsum(t_s)[-length(t_s)])
    cum_tpi <- c(0, cumsum(t_s * pi_s)[-length(t_s)])
    return(sum(t_s * (pi_s * cum_t - cum_tpi)) / (T_^2 * mu * (1 - mu)))
  }

  if (code == "Iso") {
    return(sum(t * pi^2) / (T_ * mu))
  }

  stop(sprintf("Internal: unknown index code '%s'.", code), call. = FALSE)
}


# -----------------------------------------------------------------------------
# Public API: one general entry point plus named convenience wrappers.
# -----------------------------------------------------------------------------

# Compute any two-group segregation index, optionally with the
# composition-invariance correction.
#
# data, a, b : neighborhood data, in any of three forms (see header).
# index : 'D' / 'S' / 'H' / 'R' / 'G' / 'Iso' or matching long names.
# target : if NULL, return the standard unadjusted index; if a number in
#   (min(pi), max(pi)), return the index after reweighting to that composition.
segregation_index <- function(data, a = NULL, b = NULL,
                              index = "D", target = NULL) {
  code <- .canonical_index(index)
  pt <- .to_pi_t(data, a, b)
  pi <- pt$pi; t_ <- pt$t
  if (is.null(target)) {
    mu <- sum(t_ * pi) / sum(t_)
    return(.compute(pi, t_, mu, code))
  }
  v <- .solve_v(pi, t_, target)
  .compute(pi, t_ * .weights(pi, v), target, code)
}

# Named convenience wrappers -- same signature, fixed index.
dissimilarity <- function(data, a = NULL, b = NULL, target = NULL) {
  segregation_index(data, a, b, index = "D", target = target)
}
separation <- function(data, a = NULL, b = NULL, target = NULL) {
  segregation_index(data, a, b, index = "S", target = target)
}
entropy_index <- function(data, a = NULL, b = NULL, target = NULL) {
  segregation_index(data, a, b, index = "H", target = target)
}
hutchens_r <- function(data, a = NULL, b = NULL, target = NULL) {
  segregation_index(data, a, b, index = "R", target = target)
}
gini_index <- function(data, a = NULL, b = NULL, target = NULL) {
  segregation_index(data, a, b, index = "G", target = target)
}
isolation <- function(data, a = NULL, b = NULL, target = NULL) {
  segregation_index(data, a, b, index = "Iso", target = target)
}


# -----------------------------------------------------------------------------
# Index-as-a-function-of-composition curve (no root finding).
# -----------------------------------------------------------------------------

# Sweep the tilting parameter v over v_range and return the resulting curve.
# Pass a character vector for `index` to evaluate several indices in one sweep.
#
# Returns a data.frame with column `composition` plus one column per index
# (named by its canonical short code), sorted by composition.
composition_curve <- function(data, a = NULL, b = NULL,
                              index = "D", n_points = 200L,
                              v_range = c(-30, 30)) {
  codes <- vapply(index, .canonical_index, character(1))
  pt <- .to_pi_t(data, a, b)
  pi <- pt$pi; t_ <- pt$t
  vs <- seq(v_range[1], v_range[2], length.out = n_points)

  mus  <- numeric(n_points)
  vals <- matrix(NA_real_, nrow = n_points, ncol = length(codes))
  colnames(vals) <- codes
  for (k in seq_along(vs)) {
    tw <- t_ * .weights(pi, vs[k])
    mus[k] <- sum(tw * pi) / sum(tw)
    for (j in seq_along(codes)) {
      vals[k, j] <- .compute(pi, tw, mus[k], codes[j])
    }
  }

  ord <- order(mus)
  out <- data.frame(composition = mus[ord])
  for (j in seq_along(codes)) {
    out[[codes[j]]] <- vals[ord, j]
  }
  out
}


# -----------------------------------------------------------------------------
# Optional plotting helper.
# -----------------------------------------------------------------------------

# Population-weighted percentile bounds on composition. Returns a list with
# elements `lo` and `hi`: the compositions below which / above which
# `percentile` percent of the total population lies.
.population_bounds <- function(pi, t, percentile) {
  if (!(percentile > 0 && percentile < 50)) {
    stop(sprintf("population_percentile must be in (0, 50); got %s.", percentile),
         call. = FALSE)
  }
  ord  <- order(pi)
  pi_s <- pi[ord]
  t_s  <- t[ord]
  T_   <- sum(t_s)
  cum  <- cumsum(t_s)
  frac <- percentile / 100
  # Smallest index whose cumulative population reaches the threshold.
  idx_lo <- which(cum >= frac * T_)[1]
  idx_hi <- which(cum >= (1 - frac) * T_)[1]
  list(lo = pi_s[idx_lo], hi = pi_s[idx_hi])
}

# Plot the index's composition curve over the trustworthy range.
#
# By default the curve runs from the lower to the upper
# `population_percentile`-th population-weighted percentile of composition:
# with the default value of 10, the plot spans the central 80% of the
# population. Outside this range the data is sparse and the corrected index
# is an extrapolation.
#
# No root-finding is used. The tilting parameter v is swept uniformly over
# v_range, the resulting curve is sliced to the chosen percentile range, and
# the sweep is automatically widened if the initial range does not reach
# the bounds.
#
# Pass `population_percentile = NULL` to plot the full sweep with no slicing.
# Pass `add = TRUE` to overlay on an existing plot (uses lines/points instead
# of opening a new device). The legend is off by default.
# Extra `...` arguments are forwarded to plot()/lines().
plot_composition_curve <- function(data, a = NULL, b = NULL,
                                   index = "D",
                                   population_percentile = 10,
                                   n_points = 200L,
                                   v_range = c(-30, 30),
                                   legend = FALSE,
                                   add = FALSE,
                                   ...) {
  code <- .canonical_index(index)
  pt <- .to_pi_t(data, a, b)
  pi <- pt$pi; t_ <- pt$t

  if (is.null(population_percentile)) {
    curve <- composition_curve(data, a, b, index = code,
                               n_points = n_points, v_range = v_range)
  } else {
    bounds <- .population_bounds(pi, t_, population_percentile)
    lo <- bounds$lo; hi <- bounds$hi
    lo_v <- v_range[1]; hi_v <- v_range[2]
    for (it in seq_len(10)) {  # widen up to 10 times if the sweep falls short
      curve <- composition_curve(data, a, b, index = code,
                                 n_points = n_points,
                                 v_range = c(lo_v, hi_v))
      if (min(curve$composition) <= lo && max(curve$composition) >= hi) break
      lo_v <- lo_v * 2; hi_v <- hi_v * 2
    }
    curve <- curve[curve$composition >= lo & curve$composition <= hi, ,
                   drop = FALSE]
  }

  mu0  <- sum(t_ * pi) / sum(t_)
  val0 <- .compute(pi, t_, mu0, code)

  name <- .INDEX_LONG_NAMES[code]

  if (!add) {
    plot(curve$composition, curve[[code]], type = "l",
         xlab = "Composition", ylab = name,
         main = sprintf("%s vs. composition", name),
         ...)
    grid(col = "gray85")
  } else {
    lines(curve$composition, curve[[code]], ...)
  }
  points(mu0, val0, pch = 19, col = "red", cex = 1.2)

  if (legend) {
    legend("topright",
           legend = c("Adjusted (I-projection)",
                      sprintf("Observed (composition = %.3f)", mu0)),
           lty = c(1, NA), pch = c(NA, 19),
           col = c(par("col"), "red"), bty = "n")
  }

  invisible(curve)
}


# -----------------------------------------------------------------------------
# Example (run only when this file is executed as a script).
# -----------------------------------------------------------------------------

if (sys.nframe() == 0L) {
  # Eight synthetic neighborhoods, two groups.
  df <- data.frame(
    white    = c(400, 600, 800, 200, 500, 350, 750, 100),
    hispanic = c(600, 400, 200, 800, 500, 650, 250, 900)
  )
  A <- df$white
  B <- df$hispanic
  M <- cbind(A, B)

  cat("Three equivalent input forms:\n")
  cat(sprintf("  data.frame:  D = %.4f\n", dissimilarity(df, "white", "hispanic")))
  cat(sprintf("  matrix:      D = %.4f\n", dissimilarity(M)))
  cat(sprintf("  vectors:     D = %.4f\n", dissimilarity(A, B)))

  cat("\nUnadjusted indices (white vs. hispanic):\n")
  for (code in c("D", "S", "H", "R", "G", "Iso")) {
    v <- segregation_index(df, "white", "hispanic", index = code)
    cat(sprintf("  %3s = %.4f\n", code, v))
  }

  cat("\nAdjusted to mu = 0.50:\n")
  for (code in c("D", "S", "H", "R", "G")) {
    v <- segregation_index(df, "white", "hispanic", index = code, target = 0.50)
    cat(sprintf("  %3s = %.4f\n", code, v))
  }

  cat("\nName aliasing:\n")
  cat(sprintf("  index='D'             -> %.4f\n",
              segregation_index(df, "white", "hispanic", index = "D")))
  cat(sprintf("  index='dissimilarity' -> %.4f\n",
              segregation_index(df, "white", "hispanic", index = "dissimilarity")))
  cat(sprintf("  index='Gini'          -> %.4f\n",
              segregation_index(df, "white", "hispanic", index = "Gini")))

  curve <- composition_curve(df, "white", "hispanic", index = c("D", "H", "G"))
  cat(sprintf("\nCurve: %d points, columns = %s\n",
              nrow(curve), paste(names(curve), collapse = ", ")))
  print(head(round(curve, 4), 3))
}
