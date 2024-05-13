#if !defined(__NEK5000_PARTITIONER_H__)
#define __NEK5000_PARTITIONER_H__

#include "gslib.h"

#define MAXNV 8 /* maximum number of vertices per element */

typedef struct {
  long long vtx[MAXNV];
  ulong     eid;
  int       proc;
} edata;

int parMETIS_partMesh(int *part, long long *vl, int nel, int nv, int *opt,
                      MPI_Comm ce);

#endif // __NEK5000_PARTITIONER_H__