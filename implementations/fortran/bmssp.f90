module bmssp_module
  implicit none
  integer, parameter :: MAX_NODES = 100
  integer, parameter :: MAX_EDGES = 100

  type :: Edge
    integer :: to
    real :: w
  end type Edge

  type :: Graph
    integer :: n
    integer :: deg(MAX_NODES)
    type(Edge) :: edges(MAX_NODES, MAX_EDGES)
  end type Graph

  type :: NodeDist
    integer :: node
    real :: dist
  end type NodeDist

  type :: DQueue
    integer :: M
    real :: B
    integer :: size
    type(NodeDist) :: data(MAX_NODES)
    integer :: pre_size
    type(NodeDist) :: prepend(MAX_NODES)
  end type DQueue

contains

  subroutine init_graph(g, n)
    type(Graph), intent(out) :: g
    integer, intent(in) :: n
    g%n = n
    g%deg = 0
  end subroutine init_graph

  subroutine add_edge(g, u, v, wt)
    type(Graph), intent(inout) :: g
    integer, intent(in) :: u, v
    real, intent(in) :: wt
    g%deg(u) = g%deg(u) + 1
    g%edges(u, g%deg(u)) = Edge(v, wt)
    if (g%deg(v) == 0) g%deg(v) = g%deg(v)
  end subroutine add_edge

  subroutine dq_init(q, M, B)
    type(DQueue), intent(out) :: q
    integer, intent(in) :: M
    real, intent(in) :: B
    q%M = M
    q%B = B
    q%size = 0
    q%pre_size = 0
  end subroutine dq_init

  subroutine dq_insert(q, node, dist)
    type(DQueue), intent(inout) :: q
    integer, intent(in) :: node
    real, intent(in) :: dist
    if (dist >= q%B) return
    q%size = q%size + 1
    q%data(q%size) = NodeDist(node, dist)
  end subroutine dq_insert

  subroutine dq_batch_prepend(q, items, count)
    type(DQueue), intent(inout) :: q
    type(NodeDist), intent(in) :: items(:)
    integer, intent(in) :: count
    integer :: i
    do i = 1, count
      if (items(i)%dist < q%B) then
        q%pre_size = q%pre_size + 1
        q%prepend(q%pre_size) = items(i)
      end if
    end do
  end subroutine dq_batch_prepend

  logical function dq_non_empty(q)
    type(DQueue), intent(in) :: q
    dq_non_empty = (q%pre_size > 0) .or. (q%size > 0)
  end function dq_non_empty

  subroutine dq_pull(q, Bi, nodes, count)
    type(DQueue), intent(inout) :: q
    real, intent(out) :: Bi
    integer, intent(out) :: nodes(:)
    integer, intent(out) :: count
    integer :: i, min_idx
    real :: min_dist

    if (q%pre_size > 0) then
      count = min(q%M, q%pre_size)
      Bi = q%prepend(1)%dist
      do i = 1, count
        nodes(i) = q%prepend(i)%node
      end do
      do i = 1, q%pre_size - count
        q%prepend(i) = q%prepend(i + count)
      end do
      q%pre_size = q%pre_size - count
      return
    end if

    if (q%size == 0) then
      Bi = q%B
      count = 0
      return
    end if

    min_dist = q%data(1)%dist
    min_idx = 1
    do i = 2, q%size
      if (q%data(i)%dist < min_dist) then
        min_dist = q%data(i)%dist
        min_idx = i
      end if
    end do
    Bi = min_dist
    nodes(1) = q%data(min_idx)%node
    count = 1
    do i = min_idx, q%size - 1
      q%data(i) = q%data(i + 1)
    end do
    q%size = q%size - 1
  end subroutine dq_pull

  subroutine dijkstra(g, source, dist)
    type(Graph), intent(in) :: g
    integer, intent(in) :: source
    real, intent(out) :: dist(:)
    type(DQueue) :: q
    real :: Bi
    integer :: S_nodes(MAX_NODES)
    integer :: S_count
    integer :: u, i, v
    real :: w, alt
    integer :: n

    n = g%n
    dist(1:n) = huge(0.0)
    dist(source) = 0.0

    call dq_init(q, n, huge(0.0))
    call dq_insert(q, source, 0.0)

    do while (dq_non_empty(q))
      call dq_pull(q, Bi, S_nodes, S_count)
      do i = 1, S_count
        u = S_nodes(i)
        do v = 1, g%deg(u)
          w = g%edges(u, v)%w
          alt = dist(u) + w
          if (alt < dist(g%edges(u, v)%to)) then
            dist(g%edges(u, v)%to) = alt
            call dq_insert(q, g%edges(u, v)%to, alt)
          end if
        end do
      end do
    end do
  end subroutine dijkstra

  recursive subroutine bmssp(g, l, B, S, S_count, d_hat)
    type(Graph), intent(in) :: g
    integer, intent(in) :: l
    real, intent(in) :: B
    integer, intent(in) :: S(:)
    integer, intent(in) :: S_count
    real, intent(out) :: d_hat(:)
    if (l <= 0) then
      if (S_count > 0) call dijkstra(g, S(1), d_hat)
    else
      call bmssp(g, l - 1, B, S, S_count, d_hat)
    end if
  end subroutine bmssp

  subroutine run_sssp(g, source, dist)
    type(Graph), intent(in) :: g
    integer, intent(in) :: source
    real, intent(out) :: dist(:)
    integer :: S(1)
    S(1) = source
    call bmssp(g, 1, huge(0.0), S, 1, dist)
  end subroutine run_sssp

end module bmssp_module

program main
  use bmssp_module
  implicit none
  integer, parameter :: n = 4
  type(Graph) :: g
  real :: dist(n)
  integer :: i

  call init_graph(g, n)
  call add_edge(g, 1, 2, 1.0)
  call add_edge(g, 1, 3, 4.0)
  call add_edge(g, 2, 3, 2.0)
  call add_edge(g, 2, 4, 5.0)
  call add_edge(g, 3, 4, 1.0)

  call run_sssp(g, 1, dist)

  do i = 1, n
    if (dist(i) > huge(0.0)/2) then
      print *, 'Inf'
    else
      print *, dist(i)
    end if
  end do
end program main
