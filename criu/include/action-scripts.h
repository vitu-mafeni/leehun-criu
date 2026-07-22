#ifndef __CR_ACTION_SCRIPTS_H__
#define __CR_ACTION_SCRIPTS_H__

#include <stdbool.h>

#include "asm/int.h"

enum script_actions {
	ACT_PRE_STREAM,
	ACT_PRE_DUMP,
	ACT_POST_DUMP,
	ACT_PRE_RESTORE,
	ACT_POST_RESTORE,
	ACT_NET_LOCK,
	ACT_NET_UNLOCK,
	ACT_SETUP_NS,
	ACT_POST_SETUP_NS,
	ACT_POST_RESUME,
	ACT_PRE_RESUME,
	ACT_ORPHAN_PTS_MASTER,
	ACT_STATUS_READY,
	ACT_QUERY_EXT_FILES,

	ACT_MAX
};

extern int add_script(char *path);
extern int add_rpc_notify(int sk);
extern int run_scripts(enum script_actions);
/*
 * Returns true if at least one action script (shell or RPC) is registered
 * for this restore. Used by callers that skip some restore step on the
 * assumption an external action script will handle it instead (e.g.
 * criu/mount.c's is_nvidia_dev_mount() skip) — with nothing registered at
 * all, that assumption is certainly false and callers should warn loudly
 * rather than silently proceed as if the skip were harmless.
 */
extern bool action_scripts_registered(void);
extern int rpc_send_fd(enum script_actions, int fd);
extern int rpc_query_external_files(void);
extern int exec_rpc_query_external_files(char *name, int sk);
extern int send_criu_rpc_script(enum script_actions act, char *name, int sk, int fd);

#endif /* __CR_ACTION_SCRIPTS_H__ */
