# --- Usage ---
#
# Run `nix develop` in the root project folder

{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3;
      in
      {
        devShell = pkgs.mkShell {
          packages = [
            python
            pkgs.virtualenv
          ];

          # ---- Automatically set up virtual environment ----
          shellHook =
            let
              virtualEnvironmentDirectory = ".venv";
            in
            ''
              if [ ! -d "${virtualEnvironmentDirectory}" ]; then
                echo "Virtual environment (located in the ${virtualEnvironmentDirectory} directory) does not exist."
                echo "Creating virtual environment in the ${virtualEnvironmentDirectory} directory..."
                virtualenv -p="$(which python)" "${virtualEnvironmentDirectory}"
                echo "Virtual environment created in the ${virtualEnvironmentDirectory} diretory."

                # Enter virtual environment and install packages
                source "${virtualEnvironmentDirectory}/bin/activate"
                pip install -r requirements.txt
              else
                echo "Entering the virtual environment located in the .venv directory"
                source "${virtualEnvironmentDirectory}/bin/activate"
              fi
            '';

          # ---- For dynamically linking libraries ----
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (
            with pkgs;
            [
              zlib
              zstd
              stdenv.cc.cc
              curl
              openssl
              attr
              libssh
              bzip2
              libxml2
              acl
              libsodium
              util-linux
              xz
              systemd
              libGL
              glib
            ]
          );
        };
      }
    );
}
