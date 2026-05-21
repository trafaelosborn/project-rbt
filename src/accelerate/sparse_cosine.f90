! Sparse cosine similarity kernels for Project RBT incremental scoring.
!
! Purpose:
!   The Python incremental scorer calls _sparse_profile_cosine() N times per
!   proposal (N=16 candidates × 3 profiles = 48 calls per proposal). Each call
!   iterates over a sparse dict intersection in pure Python.
!
!   These kernels replace that inner loop. The Python caller converts sparse
!   dicts to dense float32 arrays (pre-indexed against a fixed vocabulary),
!   then calls here for the dot-product / norm computation.
!
!   The vocabulary index is fixed at build time (top-N char bigrams, trigrams,
!   suffixes, word bigrams, trigrams). The caller maintains the index and
!   converts candidate counter deltas to dense delta arrays before calling.
!
! Routines:
!   cosine_f32        - cosine similarity between two dense float32 vectors
!   cosine_f64        - cosine similarity between two dense float64 vectors
!   batch_cosines_f32 - N cosines: one query vector vs one reference vector,
!                       applied to N query vectors stacked row-wise
!   dot_norm_f32      - returns dot product and both L2 norms (caller computes
!                       cosine to avoid divide-by-zero issues in Python)

module sparse_cosine_kernels
    implicit none
    integer, parameter :: sp = selected_real_kind(6, 37)   ! float32
    integer, parameter :: dp = selected_real_kind(15, 307)  ! float64

contains

    ! ---------------------------------------------------------------------------
    ! cosine_f32: cosine similarity between two dense float32 vectors
    ! ---------------------------------------------------------------------------
    subroutine cosine_f32(a, b, n, result)
        integer,  intent(in)  :: n
        real(sp), intent(in)  :: a(n), b(n)
        real(sp), intent(out) :: result
        real(sp) :: dot, norm_a, norm_b
        integer :: i

        dot    = 0.0_sp
        norm_a = 0.0_sp
        norm_b = 0.0_sp

        do i = 1, n
            dot    = dot    + a(i) * b(i)
            norm_a = norm_a + a(i) * a(i)
            norm_b = norm_b + b(i) * b(i)
        end do

        norm_a = sqrt(norm_a)
        norm_b = sqrt(norm_b)

        if (norm_a > 0.0_sp .and. norm_b > 0.0_sp) then
            result = dot / (norm_a * norm_b)
        else
            result = 0.0_sp
        end if
    end subroutine cosine_f32

    ! ---------------------------------------------------------------------------
    ! cosine_f64: cosine similarity between two dense float64 vectors
    ! ---------------------------------------------------------------------------
    subroutine cosine_f64(a, b, n, result)
        integer,  intent(in)  :: n
        real(dp), intent(in)  :: a(n), b(n)
        real(dp), intent(out) :: result
        real(dp) :: dot, norm_a, norm_b
        integer :: i

        dot    = 0.0_dp
        norm_a = 0.0_dp
        norm_b = 0.0_dp

        do i = 1, n
            dot    = dot    + a(i) * b(i)
            norm_a = norm_a + a(i) * a(i)
            norm_b = norm_b + b(i) * b(i)
        end do

        norm_a = sqrt(norm_a)
        norm_b = sqrt(norm_b)

        if (norm_a > 0.0_dp .and. norm_b > 0.0_dp) then
            result = dot / (norm_a * norm_b)
        else
            result = 0.0_dp
        end if
    end subroutine cosine_f64

    ! ---------------------------------------------------------------------------
    ! batch_cosines_f32: compute cosine(queries[i,:], ref) for i = 1..m
    !   queries : (m, n) float32 — m candidate vectors
    !   ref     : (n,)   float32 — one reference vector
    !   results : (m,)   float32 — output cosines
    ! ---------------------------------------------------------------------------
    subroutine batch_cosines_f32(queries, ref, m, n, results)
        integer,  intent(in)  :: m, n
        real(sp), intent(in)  :: queries(m, n), ref(n)
        real(sp), intent(out) :: results(m)
        real(sp) :: dot, norm_q, norm_r
        integer :: i, j

        ! Precompute reference norm (shared across all queries)
        norm_r = 0.0_sp
        do j = 1, n
            norm_r = norm_r + ref(j) * ref(j)
        end do
        norm_r = sqrt(norm_r)

        do i = 1, m
            dot    = 0.0_sp
            norm_q = 0.0_sp
            do j = 1, n
                dot    = dot    + queries(i, j) * ref(j)
                norm_q = norm_q + queries(i, j) * queries(i, j)
            end do
            norm_q = sqrt(norm_q)
            if (norm_q > 0.0_sp .and. norm_r > 0.0_sp) then
                results(i) = dot / (norm_q * norm_r)
            else
                results(i) = 0.0_sp
            end if
        end do
    end subroutine batch_cosines_f32

    ! ---------------------------------------------------------------------------
    ! dot_norm_f32: return dot product and both L2 norms separately
    ! Caller computes cosine to handle divide-by-zero.
    ! ---------------------------------------------------------------------------
    subroutine dot_norm_f32(a, b, n, dot, norm_a, norm_b)
        integer,  intent(in)  :: n
        real(sp), intent(in)  :: a(n), b(n)
        real(sp), intent(out) :: dot, norm_a, norm_b
        integer :: i

        dot    = 0.0_sp
        norm_a = 0.0_sp
        norm_b = 0.0_sp

        do i = 1, n
            dot    = dot    + a(i) * b(i)
            norm_a = norm_a + a(i) * a(i)
            norm_b = norm_b + b(i) * b(i)
        end do

        norm_a = sqrt(norm_a)
        norm_b = sqrt(norm_b)
    end subroutine dot_norm_f32

    ! ---------------------------------------------------------------------------
    ! batch_form_scores_f32:
    !   Compute Latin form scores for m candidate profiles in one pass.
    !
    !   For each candidate i:
    !     form_score[i] = 0.40 * cosine(bg_queries[i], bg_ref)
    !                   + 0.40 * cosine(tg_queries[i], tg_ref)
    !                   + 0.20 * cosine(sfx_queries[i], sfx_ref)
    !
    !   bg_queries  : (m, n_bg)  float32
    !   tg_queries  : (m, n_tg)  float32
    !   sfx_queries : (m, n_sfx) float32
    !   bg_ref      : (n_bg,)    float32
    !   tg_ref      : (n_tg,)    float32
    !   sfx_ref     : (n_sfx,)   float32
    !   form_scores : (m,)       float32  — output
    ! ---------------------------------------------------------------------------
    subroutine batch_form_scores_f32( &
        bg_queries,  n_bg,  bg_ref,  &
        tg_queries,  n_tg,  tg_ref,  &
        sfx_queries, n_sfx, sfx_ref, &
        m, form_scores)

        integer,  intent(in)  :: m, n_bg, n_tg, n_sfx
        real(sp), intent(in)  :: bg_queries(m, n_bg),   bg_ref(n_bg)
        real(sp), intent(in)  :: tg_queries(m, n_tg),   tg_ref(n_tg)
        real(sp), intent(in)  :: sfx_queries(m, n_sfx), sfx_ref(n_sfx)
        real(sp), intent(out) :: form_scores(m)

        real(sp) :: bg_cosines(m), tg_cosines(m), sfx_cosines(m)

        call batch_cosines_f32(bg_queries,  bg_ref,  m, n_bg,  bg_cosines)
        call batch_cosines_f32(tg_queries,  tg_ref,  m, n_tg,  tg_cosines)
        call batch_cosines_f32(sfx_queries, sfx_ref, m, n_sfx, sfx_cosines)

        form_scores = 0.40_sp * bg_cosines + 0.40_sp * tg_cosines + 0.20_sp * sfx_cosines

    end subroutine batch_form_scores_f32

end module sparse_cosine_kernels
