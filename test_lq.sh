#!/bin/sh

set -eu

test_config="test_config_$$.json"
test_image="test_image.$$.gif"
mock_dir="test_mock_$$"
umask 066

cleanup() {
  rm -f "$test_config" "$test_image"
  rm -rf "$mock_dir"
}

trap cleanup EXIT

mkdir -p "$mock_dir"
cat > "$mock_dir/sitecustomize.py" <<'PY'
import json
import urllib.error
import urllib.request


def pick_response(latest_user, assistant_text):
    if "first turn" in latest_user:
        return "first reply"
    if "followup" in latest_user:
        return "saw first reply" if "first reply" in assistant_text else "followup"
    if 'source="file"' in latest_user or 'source="stdin"' in latest_user:
        return "json"
    if "[image]" in latest_user and "what kind of attachment?" in latest_user:
        return "image"
    if "what letter?" in latest_user:
        return "A"
    if "good morning" in latest_user:
        return "good morning"
    if "hello" in latest_user and "times" in latest_user:
        return "hello hello hello"
    if "hello" in latest_user:
        return "hello"
    return "ok"


class MockResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __iter__(self):
        raw = json.dumps(self._payload)
        yield f"data: {raw}\n".encode("utf-8")
        yield b"data: [DONE]\n"


def urlopen(req, timeout=120):
    body = getattr(req, "data", b"") or b""
    if body:
        payload = json.loads(body.decode("utf-8"))
    else:
        payload = {}

    messages = payload.get("messages", [])
    current_user_parts = []
    assistant_parts = []

    def extend_parts(parts, content):
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    parts.append("[image]")

    started = False
    saw_assistant = False
    for message in reversed(messages):
        role = message.get("role")
        content = message.get("content")
        if not started:
            if role == "user":
                started = True
                extend_parts(current_user_parts, content)
            continue
        if role == "assistant":
            saw_assistant = True
            extend_parts(assistant_parts, content)
            continue
        if role == "user":
            if saw_assistant:
                break
            extend_parts(current_user_parts, content)
            continue
        break

    response = {
        "object": "chat.completion",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": pick_response("\n".join(reversed(current_user_parts)), "\n".join(assistant_parts)),
                }
            }
        ],
    }
    return MockResponse(response)


urllib.request.urlopen = urlopen
PY

cat > "$test_config" <<_EOF_
{
  "defaults": {
    "model_name":"mock"
  },
  "models": [
    {
      "name":"mock",
      "api_url": "http://127.0.0.1:11434/v1",
      "model": "mock"
    }
  ],
  "templates": [
    {
      "name": "t0",
      "prompt": "say hello"
    },
    {
      "name": "t1",
      "prompt": "say %s",
      "defaults": ["hello"]
    },
    {
      "name": "t2",
      "prompt": "say %s %s times",
      "defaults": [null, 3]
    }
  ]
}
_EOF_

base64 -d <<_EOF_ > "$test_image"
R0lGODlhFAAUAPABAAAAAP///yH5BAAAAAAAIf8LSW1hZ2VNYWdpY2sOZ2FtbWE9MC40NT
Q1NDUALAAAAAAUABQAAAIyjI+pCw0Lg3tRzZpmw2dLynke5hglCX5hekLtl1qxGmm2a2v4
zsym/0IEaTyhrzPiLAoAOw==
_EOF_

lq="env PYTHONPATH=$mock_dir${PYTHONPATH:+:$PYTHONPATH} python3 lq.py -c $test_config"

# do test: T "test name" expected_result "test command"
T(){
  printf "%s: " "$1"
  expected=$2
  shift 2
  if eval "$@" >/dev/null 2>/dev/null; then
    rc=0
  else
    rc=$?
  fi
  case $rc in
  $expected) success=$((success+1)); echo ok;;
  *)         fail=$((fail+1)); echo fail;;
  esac
}

success=0
fail=0

T "simple query" 0 '$lq "say hello" </dev/null | grep -i hello'
T "file attachment" 0 '$lq -f $test_config "what type is this file?" </dev/null | grep -i json'
T "image attachment" 0 '$lq -i $test_image "what letter?" </dev/null | grep -o A'
T "read from stdin" 0 '$lq "what type is this file?" < $test_config | grep -i json'
T "no prompt" 1 '$lq -f $ test_config'
T "json output" 0 '$lq --json "say hello" </dev/null | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin)[\"object\"] == \"chat.completion\" else 1)"'
T "template without parameter" 0 '$lq -t t0 </dev/null | grep -i hello'
T "template with default parameter" 0 '$lq -t t1 </dev/null | grep -i hello'
T "template with parameter from arg" 0 '$lq -t t1 "good morning" </dev/null | grep -i "good morning"'
T "template with 2 parameters" 0 't=$($lq -t t2 hello 3 </dev/null | xargs -n1 | grep -ioc hello); [ $t -eq 3 ]'
T "template with too many params" 1 '$lq -t t1 foo bar'
T "template with too few params" 1 '$lq -t t2'
T "chat without tty" 1 '$lq --chat < $test_config'
T "ignore chat" 0 '$lq --chat "what type is this file?" < $test_config 2>&1 | grep Warning'

cat > "$mock_dir/chat.exp" <<EOF
#!/usr/bin/expect -f
set timeout 10
log_user 0
spawn sh -c "$lq --chat --no-stream"
expect {
  -re {prompt> } {}
  timeout { exit 1 }
}
send -- "/file $test_config\r"
expect {
  -re {prompt> } {}
  timeout { exit 1 }
}
send -- "what type is this file?\r"
expect {
  -re {json} {}
  timeout { exit 1 }
}
expect {
  -re {prompt> } {}
  timeout { exit 1 }
}
send -- "/image $test_image\r"
expect {
  -re {prompt> } {}
  timeout { exit 1 }
}
send -- "what kind of attachment?\r"
expect {
  -re {image} {}
  timeout { exit 1 }
}
expect {
  -re {prompt> } {}
  timeout { exit 1 }
}
send -- "/template t1 \"good morning\"\r"
expect {
  -re {prompt> } {}
  timeout { exit 1 }
}
send -- "\r"
expect {
  -re {good morning} {}
  timeout { exit 1 }
}
expect {
  -re {prompt> } {}
  timeout { exit 1 }
}
send -- "/quit\r"
expect eof
EOF
chmod +x "$mock_dir/chat.exp"
T "chat interactive" 0 "/usr/bin/expect $mock_dir/chat.exp"

chmod 644 "$test_config"
T "config permission" 0 '$lq say hello </dev/null 2>&1 | grep Warning'

echo
echo "Success: $success, Fail: $fail"
