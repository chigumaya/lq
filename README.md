# lq (LLM query)

## What is this?

A command-line tool for querying an LLM using data obtained from local files or standard input.

Currently supports only OpenAI compatible API (`POST /chat/completions`).

## Installation

Written in Python 3 with no non-standard dependencies. Simply copy it somewhere in your PATH.

```bash
install -m 755 lq.py /path/to/lq
```

## How to Use

```bash
lq [OPTIONS] <prompt>
```

### Options

```
-f, --file <filename>              Adds a file to the prompt (can be specified multiple times)
-i, --image <filename>             Adds an image to the prompt (can be specified multiple times)
-s, --system <text>                Specifies the additional system prompt as a string
-S, --system-file <filename>       Reads the additional system prompt from a file
-c, --config <path>                Path to the configuration file (Default: ~/.config/lq/config.json)
-m, --model <name>                 Specifies the model to use
-M, --max-size <size>              Maximum input size (Default: 10MB)
-j, --json                         Outputs raw JSON response
--no-stream                        Disable streaming output
--debug                            Debug mode (outputs request information to stderr)
-h, --help                         Displays help
-v, --version                      Displays version
```

## Configuration

### Configuration File

Configured in `~/.config/lq/config.json`. A different path can be specified using `-c`. Multiple models can be defined, and by default, the model specified in `defaults.model_name` is used. If you want to use a different model, specify it with `-m`.

```json
{
  "defaults": {
    "model_name": "foo",
    "max_size": "10MB"
  },
  "models": [
    {
      "name": "foo",
      "model": "foo-model",
      "api_url": "https://example.com/v1",
      "api_key": "sk-foo-key"
    },
    {
      "name": "bar",
      "model": "bar-model",
      "api_url": "https://example.net/v1",
      "api_key": "sk-bar-key"
    }
  ]
}
```

### Environment Variables

Settings equivalent to the configuration file can also be specified via environment variables.
```bash
API_URL=http://localhost:11434/v1  URL of the API endpoint
API_KEY=sk-...                     API key (optional)
MODEL=gpt-oss-20b                  Model name
```

### Priority of Configuration Values

In descending order of priority:

1. Command-line options (`-m`, `-s`, etc.)
2. Environment variables (`API_URL`, `API_KEY`, `MODEL`)
3. Configuration file (`~/.config/lq/config.json`)

## Usage Examples

```bash
# Simple question
% lq 'what is the command to extract a .tar.gz?'

# Analyze a file
% lq -f README.md 'summarize this'

# Compare multiple files
% lq -f README.md -f README.ja.md 'compare the files'

# Describe an image
% lq -i diagram.png 'explain this diagram'

# Read a QR code
% lq -i qrcode.png 'read the QR code'

# Analyze differences
% diff file-A file-B | lq 'summarize the differences'

# Analyze logs
% tail /var/log/messages | lq 'diagnose problems from these logs'

# Translate text
% man curl | lq 'translate to Japanese'

# Analyze processes
% ps auxw | lq 'which process is using the most memory?'

# Generate commit message
% git commit -m "$(git diff | lq 'summarize the difference in 3 lines')"
```

### Handling of Files (`-f`) and Images (`-i`)

If you specify an image file using `-f` instead of `-i`, like this:
```bash
% lq -f image.png 'what is this image?'
```
the file will be sent to the LLM as base64 encoded binary data, not as an image. When sent as an image, the LLM performs dedicated image processing, and token consumption is very low. However, when sent as base64 encoded data, it is treated as text data, which can result in extremely high token consumption—be aware of this.

The tool only determines if data received from standard input is text or binary; it does not determine whether the binary data can be treated as an image. Therefore, for receiving images via standard input, using process substitution with the `-i` option is more token-efficient than using a pipe (`|`).

```bash
# Treating data obtained via HTTPS as an image (process substitution)
% lq -i <(curl https://example.com/foo.jpg) 'read the text contained in the image'

# Treating data obtained via HTTPS as mere binary data (pipe)
% curl https://example.com/foo.jpg | lq 'read the text contained in the image'
```

On the other hand, even if an LLM does not support image input, sometimes you can make it recognize images by deliberately using `-f` (or a pipe) instead of `-i`.

Note that it does not support text encodings other than UTF-8.

## Security Considerations

### Prompt Injection

Currently, the LLM must treat attached data as part of the prompt, even if it is intended not to be treated as executable instructions. Therefore, `lq` instructs the system prompt to ignore any instructions contained within files, standard input, or image attachments.

However, since different LLMs interpret instructions differently, and defense can potentially be bypassed by cleverly crafted payloads, complete countermeasures are difficult. While efforts have been made to reduce the risk of unintended command execution or role-playing attacks, this is merely a risk mitigation measure and should not be considered a guarantee. You should not use `lq` to handle untrusted data.

### Information Leakage

Files passed via `-f`, images passed via `-i`, and data passed from standard input are all sent to the LLM as part of the prompt. Even if these data contain passwords, secret keys, authentication tokens, or other confidential information, they will be sent without masking. Therefore, extreme caution is advised when using cloud LLM services. For such use cases, consider using a local LLM.

## Author

<mailto:y@maya.st>

## License

MIT
