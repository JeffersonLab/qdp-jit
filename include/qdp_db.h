#ifndef QDP_DB_H
#define QDP_DB_H

/*! @file
 * @brief Support for filedb
 */

#ifdef BUILD_FILEDB

// FileDB is used, so define a real set of classes
#include "qdp_db_imp.h"

#else

// FileDB is NOT used, so define stubs of classes
#include "qdp_db_stub.h"
#endif

#endif
