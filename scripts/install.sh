#!/bin/sh
set -eu

repository="99percentgrip/Native-GLM-ACP"
release_base="${GLM_ACP_RELEASE_BASE_URL:-https://github.com/$repository/releases}"
version="${GLM_ACP_VERSION:-latest}"
install_dir="${GLM_ACP_INSTALL_DIR:-${XDG_BIN_HOME:-$HOME/.local/bin}}"

fail() {
    printf 'glm-acp installer: %s\n' "$1" >&2
    exit 1
}

command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v tar >/dev/null 2>&1 || fail "tar is required"

case "$(uname -s)" in
    Linux) platform="linux" ;;
    Darwin) platform="darwin" ;;
    *) fail "unsupported operating system: $(uname -s)" ;;
esac

case "$(uname -m)" in
    x86_64|amd64) architecture="x86_64" ;;
    arm64|aarch64) architecture="aarch64" ;;
    *) fail "unsupported architecture: $(uname -m)" ;;
esac

asset="native-glm-acp-$platform-$architecture.tar.gz"
if [ "$version" = "latest" ]; then
    download_root="$release_base/latest/download"
else
    case "$version" in
        v*) tag="$version" ;;
        *) tag="v$version" ;;
    esac
    download_root="$release_base/download/$tag"
fi

temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT HUP INT TERM

printf 'Downloading %s...\n' "$asset"
curl -fL --retry 3 --show-error --silent "$download_root/$asset" -o "$temporary/$asset"
curl -fL --retry 3 --show-error --silent "$download_root/$asset.sha256" -o "$temporary/$asset.sha256"

if command -v sha256sum >/dev/null 2>&1; then
    (cd "$temporary" && sha256sum -c "$asset.sha256")
elif command -v shasum >/dev/null 2>&1; then
    expected="$(awk '{print $1}' "$temporary/$asset.sha256")"
    actual="$(shasum -a 256 "$temporary/$asset" | awk '{print $1}')"
    [ "$actual" = "$expected" ] || fail "SHA-256 verification failed"
else
    fail "sha256sum or shasum is required to verify the download"
fi

tar -xzf "$temporary/$asset" -C "$temporary"
[ -f "$temporary/native-glm-acp" ] || fail "archive did not contain native-glm-acp"

mkdir -p "$install_dir"
install -m 0755 "$temporary/native-glm-acp" "$install_dir/native-glm-acp"
ln -sf native-glm-acp "$install_dir/glm-acp"

installed_version="$($install_dir/native-glm-acp --version)"
printf 'Installed Native GLM ACP %s:\n' "$installed_version"
printf '  %s\n' "$install_dir/native-glm-acp"
printf '  %s\n' "$install_dir/glm-acp"

case ":${PATH:-}:" in
    *":$install_dir:"*) ;;
    *)
        if [ -n "${GLM_ACP_SHELL_PROFILE:-}" ]; then
            shell_profile="$GLM_ACP_SHELL_PROFILE"
        else
            case "${SHELL:-}" in
                */zsh) shell_profile="$HOME/.zprofile" ;;
                *) shell_profile="$HOME/.profile" ;;
            esac
        fi
        path_line="export PATH=\"$install_dir:\$PATH\""
        if [ ! -f "$shell_profile" ] || ! grep -Fqx "$path_line" "$shell_profile"; then
            printf '\n# Native GLM ACP\n%s\n' "$path_line" >> "$shell_profile"
        fi
        printf '\nAdded %s to PATH in %s. Open a new terminal to use it.\n' \
            "$install_dir" "$shell_profile"
        ;;
esac

printf '\nNext: glm-acp --setup\n'
printf 'Then start the full-screen agent: glm-acp chat\n'
printf 'Inside the TUI, type / for the live menu, follow the animated status line, and press Ctrl-X to exit.\n'
