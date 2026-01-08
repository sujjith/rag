"""
Phase 1.1: Verify Kubeflow Installation

This script checks that Kubeflow is properly installed and accessible.

Prerequisites:
- Kubeflow installed on Minikube
- Port forwarding active (./scripts/04_port_forward.sh)

Run: python 01_verify_installation.py
"""
import subprocess
import sys

def run_command(cmd):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        return False, str(e)


def check_kubectl():
    """Check kubectl is available."""
    success, output = run_command("kubectl version --client --short")
    return success, output


def check_minikube():
    """Check Minikube status."""
    success, output = run_command("minikube status")
    return "Running" in output, output


def check_kubeflow_namespace():
    """Check kubeflow namespace exists."""
    success, output = run_command("kubectl get namespace kubeflow")
    return success, output


def check_pods(namespace):
    """Check pods in namespace."""
    cmd = f"kubectl get pods -n {namespace} --no-headers"
    success, output = run_command(cmd)
    if not success:
        return 0, 0, []

    lines = [l for l in output.split('\n') if l.strip()]
    total = len(lines)
    running = sum(1 for l in lines if 'Running' in l or 'Completed' in l)

    return total, running, lines


def check_http_endpoint(url):
    """Check if HTTP endpoint is reachable."""
    try:
        import urllib.request
        req = urllib.request.urlopen(url, timeout=5)
        return req.status == 200
    except:
        return False


def main():
    print("=" * 60)
    print("Kubeflow Installation Verification")
    print("=" * 60)

    checks = []

    # Check 1: kubectl
    print("\n[1/6] Checking kubectl...")
    success, msg = check_kubectl()
    checks.append(("kubectl", success))
    print(f"      {'✓' if success else '✗'} {msg if success else 'kubectl not found'}")

    # Check 2: Minikube
    print("\n[2/6] Checking Minikube...")
    success, msg = check_minikube()
    checks.append(("Minikube", success))
    print(f"      {'✓' if success else '✗'} {'Running' if success else 'Not running'}")

    # Check 3: Kubeflow namespace
    print("\n[3/6] Checking Kubeflow namespace...")
    success, _ = check_kubeflow_namespace()
    checks.append(("Namespace", success))
    print(f"      {'✓' if success else '✗'} kubeflow namespace {'exists' if success else 'not found'}")

    # Check 4: Kubeflow pods
    print("\n[4/6] Checking Kubeflow pods...")
    total, running, _ = check_pods("kubeflow")
    success = running > 0 and running >= total * 0.8
    checks.append(("Pods", success))
    print(f"      {'✓' if success else '✗'} {running}/{total} pods running")

    # Check 5: Istio pods
    print("\n[5/6] Checking Istio pods...")
    total, running, _ = check_pods("istio-system")
    success = running > 0
    checks.append(("Istio", success))
    print(f"      {'✓' if success else '✗'} {running}/{total} pods running")

    # Check 6: Dashboard endpoint
    print("\n[6/6] Checking Dashboard endpoint...")
    success = check_http_endpoint("http://localhost:8080")
    checks.append(("Dashboard", success))
    print(f"      {'✓' if success else '✗'} http://localhost:8080 {'reachable' if success else 'not reachable'}")
    if not success:
        print("      Run: ./scripts/04_port_forward.sh")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, s in checks if s)
    total = len(checks)

    for name, success in checks:
        print(f"  {'✓' if success else '✗'} {name}")

    print(f"\n  Result: {passed}/{total} checks passed")

    if passed == total:
        print("\n  Kubeflow is ready to use!")
        print("\n  Dashboard: http://localhost:8080")
        print("  Credentials: user@example.com / 12341234")
    else:
        print("\n  Some checks failed. Please review the installation.")

    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
