subroutine elementwise_abs_distance(current, reference, out, n_rows, n_cols)
    implicit none

    integer, intent(in) :: n_rows, n_cols
!f2py intent(hide), depend(current) :: n_rows = shape(current, 0)
!f2py intent(hide), depend(current) :: n_cols = shape(current, 1)
    real(8), intent(in) :: current(n_rows, n_cols)
    real(8), intent(in) :: reference(n_rows, n_cols)
    real(8), intent(out) :: out(n_rows, n_cols)

    integer :: i, j

    do j = 1, n_cols
        do i = 1, n_rows
            out(i, j) = abs(current(i, j) - reference(i, j))
        end do
    end do
end subroutine elementwise_abs_distance

subroutine insert_candidate( &
    component_id, row_index, col_index, signed_delta, abs_score, &
    max_k, actual_k, component_ids, row_indices, col_indices, signed_deltas, abs_scores)

    implicit none

    integer, intent(in) :: component_id, row_index, col_index, max_k
    real(8), intent(in) :: signed_delta, abs_score
    integer, intent(inout) :: actual_k
    integer, intent(inout) :: component_ids(max_k)
    integer, intent(inout) :: row_indices(max_k)
    integer, intent(inout) :: col_indices(max_k)
    real(8), intent(inout) :: signed_deltas(max_k)
    real(8), intent(inout) :: abs_scores(max_k)

    integer :: insert_pos

    if (actual_k < max_k) then
        actual_k = actual_k + 1
        insert_pos = actual_k
    else
        if (abs_score <= abs_scores(max_k)) then
            return
        end if
        insert_pos = max_k
    end if

    do while (insert_pos > 1 .and. abs_score > abs_scores(insert_pos - 1))
        component_ids(insert_pos) = component_ids(insert_pos - 1)
        row_indices(insert_pos) = row_indices(insert_pos - 1)
        col_indices(insert_pos) = col_indices(insert_pos - 1)
        signed_deltas(insert_pos) = signed_deltas(insert_pos - 1)
        abs_scores(insert_pos) = abs_scores(insert_pos - 1)
        insert_pos = insert_pos - 1
    end do

    component_ids(insert_pos) = component_id
    row_indices(insert_pos) = row_index
    col_indices(insert_pos) = col_index
    signed_deltas(insert_pos) = signed_delta
    abs_scores(insert_pos) = abs_score
end subroutine insert_candidate

subroutine top_adjustments_batch( &
    cooc_current, cooc_reference, &
    pos_current, pos_reference, &
    max_k, &
    component_ids, row_indices, col_indices, signed_deltas, abs_scores, actual_k, &
    n_vocab, pos_width)

    implicit none

    integer, intent(in) :: max_k, n_vocab, pos_width
!f2py intent(hide), depend(cooc_current) :: n_vocab = shape(cooc_current, 0)
!f2py intent(hide), depend(pos_current) :: pos_width = shape(pos_current, 1)
    real(8), intent(in) :: cooc_current(n_vocab, n_vocab)
    real(8), intent(in) :: cooc_reference(n_vocab, n_vocab)
    real(8), intent(in) :: pos_current(n_vocab, pos_width)
    real(8), intent(in) :: pos_reference(n_vocab, pos_width)
    integer, intent(out) :: component_ids(max_k)
    integer, intent(out) :: row_indices(max_k)
    integer, intent(out) :: col_indices(max_k)
    real(8), intent(out) :: signed_deltas(max_k)
    real(8), intent(out) :: abs_scores(max_k)
    integer, intent(out) :: actual_k

    integer :: i, j
    real(8) :: delta, score

    component_ids = 0
    row_indices = 0
    col_indices = 0
    signed_deltas = 0.0d0
    abs_scores = -1.0d0
    actual_k = 0

    do j = 1, n_vocab
        do i = 1, n_vocab
            delta = cooc_reference(i, j) - cooc_current(i, j)
            score = abs(delta)
            call insert_candidate( &
                1, i, j, delta, score, &
                max_k, actual_k, component_ids, row_indices, col_indices, signed_deltas, abs_scores)
        end do
    end do

    do j = 1, pos_width
        do i = 1, n_vocab
            delta = pos_reference(i, j) - pos_current(i, j)
            score = abs(delta)
            call insert_candidate( &
                2, i, j, delta, score, &
                max_k, actual_k, component_ids, row_indices, col_indices, signed_deltas, abs_scores)
        end do
    end do

end subroutine top_adjustments_batch
