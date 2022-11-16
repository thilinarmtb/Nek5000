#ifndef CRS_H
#define CRS_H

#if defined(PARRSB)
#include "coarse.h"
#endif

#if !defined(COMM_H)
#warning "crs.h" requires "comm.h"
#endif

#define crs_xxt_setup PREFIXED_NAME(crs_xxt_setup)
#define crs_xxt_solve PREFIXED_NAME(crs_xxt_solve)
#define crs_xxt_stats PREFIXED_NAME(crs_xxt_stats)
#define crs_xxt_free  PREFIXED_NAME(crs_xxt_free )

#define crs_amg_setup PREFIXED_NAME(crs_amg_setup)
#define crs_amg_solve PREFIXED_NAME(crs_amg_solve)
#define crs_amg_stats PREFIXED_NAME(crs_amg_stats)
#define crs_amg_free  PREFIXED_NAME(crs_amg_free )

struct coarse *crs_xxt_setup(
  uint n, const ulong *id,
  uint nz, const uint *Ai, const uint *Aj, const double *A,
  uint null_space, const struct comm *comm);
void crs_xxt_solve(double *x, struct coarse *data, double *b);
void crs_xxt_stats(struct coarse *data);
void crs_xxt_free(struct coarse *data);

struct coarse *crs_amg_setup(
  uint n, const ulong *id,
  uint nz, const uint *Ai, const uint *Aj, const double *A,
  uint null_space, const struct comm *comm,
  const char *datafname, uint *ierr);
void crs_amg_solve(double *x, struct coarse *data, double *b);
void crs_amg_stats(struct coarse *data);
void crs_amg_free(struct coarse *data);

struct coarse *ccrs_hypre_setup(
  uint n, const ulong *id,
  uint nz, const uint *Ai, const uint *Aj, const double *Av, 
  const uint nullspace, const struct comm *comm,
  const double *param);
void ccrs_hypre_solve(double *x, struct coarse *data, double *b);
void ccrs_hypre_free(struct coarse *data);

#endif
