#!/bin/bash
# TEMPORARY (debugging stuck Read the Docs builds, remove when resolved).
#
# Read the Docs only stores the output of a build command once it exits, so a
# build that gets stuck leaves no trace. This script reports from inside the
# build container to an ntfy.sh topic instead:
#
#   rtd_watchdog.sh info    print (and send) details about the build machine
#   rtd_watchdog.sh start   start the reporter in the background and return
#   rtd_watchdog.sh sphinx  run the Sphinx HTML build, also logging to $LOG
#
# (The Sphinx call lives here because Read the Docs runs job commands as
# `/bin/sh -c '<command>'` without escaping, so a command cannot contain
# single quotes, and /bin/sh is dash, which has no `pipefail`.)
#
# The reporter sends a status line every $INTERVAL seconds (container memory,
# pressure, processes, last lines of the Sphinx log). When the Sphinx log has
# not grown for $STALL seconds, it asks every Python process that handles
# SIGUSR1 to dump the stacks of all its threads (via faulthandler, into the
# Sphinx log) and uploads the log tail plus per-thread kernel states.
#
# Messages can be read with:  curl -s "https://ntfy.sh/$TOPIC/json?poll=1&since=all"

TOPIC=${MANIM_RTD_NTFY_TOPIC:-manim-rtd-8c9bdf1137dc3dbc}
URL="https://ntfy.sh/$TOPIC"
LOG=${MANIM_RTD_SPHINX_LOG:-/tmp/sphinx.log}
INTERVAL=${MANIM_RTD_WATCHDOG_INTERVAL:-30}
STALL=${MANIM_RTD_WATCHDOG_STALL:-180}
MAX_RUNTIME=${MANIM_RTD_WATCHDOG_MAX_RUNTIME:-2700}
BUILD="${READTHEDOCS_VERSION:-local}@${READTHEDOCS_GIT_COMMIT_HASH:0:8}"

send() {  # send <title> <message>
    printf '%s' "$2" | head -c 3900 |
        curl -s --max-time 10 -H "Title: $1" --data-binary @- "$URL" >/dev/null 2>&1
}

attach() {  # attach <title> <file name> <path>
    curl -s --max-time 30 -H "Title: $1" -H "Filename: $2" -T "$3" "$URL" >/dev/null 2>&1
}

cgroup_file() {  # cgroup v2 first, then v1
    for f in "/sys/fs/cgroup/$1" "/sys/fs/cgroup/memory/$2"; do
        [ -r "$f" ] && { cat "$f"; return; }
    done
    echo "n/a"
}

machine_info() {
    echo "== kernel";    uname -a; cat /proc/version
    echo "== cpus";      nproc; grep -m1 "model name" /proc/cpuinfo; grep -c ^processor /proc/cpuinfo
    echo "== memory";    free -m
    echo "== cgroup";    echo "memory.max=$(cgroup_file memory.max memory.limit_in_bytes)"
                         echo "cpu.max=$(cgroup_file cpu.max cpu.cfs_quota_us)"
                         echo "pids.max=$(cgroup_file pids.max pids.max)"
                         cat /proc/self/cgroup
    echo "== pid 1";     tr '\0' ' ' < /proc/1/cmdline; echo
    echo "== dmesg";     dmesg 2>&1 | head -n 20
    echo "== mounts";    head -n 15 /proc/self/mounts
    echo "== limits";    ulimit -a
    echo "== disk";      df -h /home/docs /tmp 2>&1
}

status() {
    local now=$1
    echo "t=${now}s log=$(stat -c %s "$LOG" 2>/dev/null || echo -)B" \
        "mem=$(( $(cgroup_file memory.current memory.usage_in_bytes 2>/dev/null || echo 0) / 1048576 ))MB" \
        "threads=$(ps -eLo tid= | wc -l) load=$(cut -d' ' -f1-3 /proc/loadavg)"
    echo "memory.events: $(cgroup_file memory.events memory.oom_control | tr '\n' ' ')"
    for p in memory cpu io; do
        echo "$p.pressure: $(head -n1 "/sys/fs/cgroup/$p.pressure" 2>/dev/null)"
    done
    ps -eo pid,ppid,stat,etimes,rss,nlwp,wchan:22,args --sort=-rss |
        grep -v -e healthcheck -e "ps -eo" -e "rtd_watchdog" | cut -c1-180 | head -n 12
    echo "-- log tail"
    tail -n 4 "$LOG" 2>/dev/null | cut -c1-300
}

dump_stacks() {
    local out=/tmp/watchdog-dump.txt
    # SIGUSR1 only goes to processes that handle it; for the others, the
    # default action would be to terminate them.
    for status_file in /proc/[0-9]*/status; do
        pid=${status_file#/proc/}; pid=${pid%/status}
        caught=$(awk '/^SigCgt:/ {print $2}' "$status_file" 2>/dev/null)
        [ -n "$caught" ] && (( 0x$caught & 0x200 )) && kill -USR1 "$pid" 2>/dev/null
    done
    sleep 5
    {
        status "$1"
        echo; echo "== threads (kernel state)"
        ps -eLo pid,tid,stat,etimes,pcpu,wchan:32,comm | grep -v -e "ps -eLo"
        echo; echo "== memory.stat"; cgroup_file memory.stat memory.stat
        echo; echo "== sphinx log (tail, including Python stack dumps)"
        tail -n 400 "$LOG" 2>/dev/null
    } > "$out" 2>&1
    attach "STALLED $BUILD t=${1}s" "dump-${1}s.txt" "$out"
}

loop() {
    local start last_size=-1 last_change last_dump=-100000 now size
    start=$(date +%s); last_change=$start
    send "watchdog started $BUILD" "$(machine_info 2>&1 | head -c 3800)"
    while :; do
        sleep "$INTERVAL"
        now=$(( $(date +%s) - start ))
        size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
        if [ "$size" != "$last_size" ]; then
            last_size=$size; last_change=$(date +%s)
        fi
        send "$BUILD t=${now}s" "$(status "$now" 2>&1)"
        if (( $(date +%s) - last_change >= STALL && now - last_dump >= 300 )); then
            dump_stacks "$now"; last_dump=$now
        fi
        (( now >= MAX_RUNTIME )) && break
    done
}

case "$1" in
    info)
        machine_info 2>&1 | tee /tmp/watchdog-info.txt
        attach "machine info $BUILD" "machine-info.txt" /tmp/watchdog-info.txt
        ;;
    start)
        # Fully detached, so the `docker exec` running this command returns.
        setsid nohup bash "$0" loop < /dev/null > /tmp/watchdog.out 2>&1 &
        echo "watchdog started, reporting to $URL"
        ;;
    loop)
        loop
        ;;
    sphinx)
        cd docs/source || exit 1
        set -o pipefail
        python -u -X faulthandler -m sphinx -T -v -b html -d _build/doctrees \
            -D language=en . "$READTHEDOCS_OUTPUT/html" 2>&1 | tee "$LOG"
        ;;
    *)
        echo "usage: $0 info|start|sphinx" >&2
        exit 2
        ;;
esac
