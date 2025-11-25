from kubernetes import client, config

# from mcp.server.fastmcp import FastMCP      │ 4 lines yanked                       │
from fastmcp import FastMCP
import logging
import subprocess
import os


mcp = FastMCP("ArgoKubernetes MCP server")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global API clients that will be reinitialized when context changes
coreV1 = None
appsV1 = None
batchV1 = None
networkingV1 = None
customObjectsApi = None

# Global variable to track current kubectl context
# current_kubectl_context = "spendor.cluster"
current_kubectl_context = "k3d-sunrise"
current_namespace_context = "argo"

# Load initial kubeconfig (usually from ~/.kube/config)
config.load_kube_config()


def initialize_clients():
    """Initialize or reinitialize all Kubernetes API clients."""
    # Create API clients
    global coreV1, appsV1, batchV1, networkingV1, customObjectsApi
    coreV1 = client.CoreV1Api()
    appsV1 = client.AppsV1Api()
    batchV1 = client.BatchV1Api()
    networkingV1 = client.NetworkingV1Api()
    customObjectsApi = client.CustomObjectsApi()
    logger.info("Kubernetes API clients initialized")


def sc(context=current_kubectl_context):
    try:
        global current_kubectl_context

        # First, switch the kubectl config context permanently
        subprocess.run(
            ["kubectl", "config", "use-context", context],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"kubectl config switched to: {context}")
        # Then load the kube config for Python client
        config.load_kube_config(context=context)
        initialize_clients()  # Reinitialize clients with new context
        current_kubectl_context = context  # Track the current context globally
        logger.info(f"Successfully switched to context: {context}")
        return {"message": f"Switched to context: {context}"}
    except subprocess.CalledProcessError as cmd_error:
        logger.error(f"Error switching kubectl context: {cmd_error.stderr}")
        return {"error": f"kubectl context switch failed: {cmd_error.stderr}"}
    except Exception as e:
        logger.error(f"Error switching context to {context}: {e}")
        return {"error": str(e)}


def sn(namespace=current_namespace_context):
    try:
        global current_namespace_context
        # First, switch the kubectl config context permanently
        subprocess.run(
            ["kubectl", "config", "set-context", "--current", "--namespace", namespace],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"kubectl config switched to: {namespace}")
        current_namespace_context = namespace  # Track the current context globally
        return {"message": f"Switched to namespace: {namespace}"}
    except subprocess.CalledProcessError as cmd_error:
        logger.error(f"Error switching kubectl context: {cmd_error.stderr}")
        return {"error": f"kubectl context switch failed: {cmd_error.stderr}"}
    except Exception as e:
        logger.error(f"Error switching namespace to {namespace}: {e}")
        return {"error": str(e)}


@mcp.tool()
def switch_context(context: str):
    """
    Switch the active Kubernetes context to connect to a different cluster.

    This tool changes the current Kubernetes context to the specified one, allowing
    you to switch between different clusters or namespaces. After switching, all
    subsequent kubectl commands and API calls will be directed to the new context.
    The Kubernetes API clients are automatically reinitialized for the new context.

    Args:
        context (str): The name of the context to switch to. Must be a valid context
                      name from your kubeconfig file. Use list_clusters() to see
                      available contexts.

    Returns:
        dict: A dictionary containing:
            - message: Success message if context switch was successful
            - error: Error message if the context switch failed

    Example:
        Input: "dev-cluster"
        Returns: {"message": "Switched to context: dev-cluster"}
    """
    sc(context)


@mcp.tool()
def switch_namespace(namespace: str):
    """
    Switch the active Kubernetes namespace to connect to a different namespace.

    This tool changes the current Kubernetes namespace to the specified one, allowing
    you to switch between different namespaces. After switching, all
    subsequent kubectl commands and API calls will be directed to the new namespace.
    The Kubernetes API clients are automatically reinitialized for the new namespace.

    Args:
        namespace (str): The name of the namespace to switch to. Must be a valid namespace
                      name from your cluster. Use list_namespaces() to see
                      available namespaces.

    Returns:
        dict: A dictionary containing:
            - message: Success message if namespace switch was successful
            - error: Error message if the namespace switch failed

    Example:
        Input: "argo"
        Returns: {"message": "Switched to namespace: argo"}
    """
    sn(namespace)


@mcp.tool()
def run_kubectl_command(command: str):
    """
    Execute any kubectl command with full privileges (use with caution).

    This tool allows execution of any kubectl command, including potentially
    destructive operations like delete, update, patch, apply, etc. It provides
    complete access to your Kubernetes cluster with the same permissions as
    your kubectl configuration.

    WARNING: This tool can perform destructive operations. Use run_kubectl_command_ro()
    for safe, read-only operations when you only need to gather information.

    Args:
        command (str): The complete kubectl command to execute. Must start with "kubectl".
                      Examples: "kubectl delete pod nginx", "kubectl apply -f config.yaml"

    Returns:
        str: The stdout output from the kubectl command, or an error message if the
             command fails or doesn't start with "kubectl".

    Example:
        Input: "kubectl scale deployment nginx --replicas=3"
        Returns: "deployment.apps/nginx scaled"
    """
    try:
        global current_kubectl_context

        # Check if command starts with kubectl
        if not command.startswith("kubectl "):
            return "Error: Command must start with 'kubectl'"

        # Add context flag if a context has been switched and --context is not already in command
        if current_kubectl_context and "--context" not in command:
            command_parts = command.split()
            command_parts.insert(1, f"--context={current_kubectl_context}")
            command = " ".join(command_parts)
            logger.info(
                f"Using context: {current_kubectl_context} for command: {command}"
            )

        # Execute the full command as provided
        result = subprocess.run(
            command.split(), capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running kubectl command: {e.stderr}")
        return f"Error: {e.stderr}"


@mcp.tool()
def run_kubectl_command_ro(command: str):
    """
    Execute read-only kubectl commands safely for information gathering.

    This tool provides a safe way to run kubectl commands that only read information
    from your Kubernetes cluster without making any modifications. It blocks potentially
    destructive operations and only allows commands that gather information.

    Allowed operations:
    - get: Retrieve resources (pods, deployments, services, etc.)
    - describe: Show detailed information about resources
    - explain: Show documentation for resource types
    - logs: Retrieve the logs of the resources (pods, deployments, services, etc.)
    - top: Display resource usage (cpu/memory)
    - config view: View kubeconfig settings
    - config get-contexts: List available contexts
    - version: Show kubectl and cluster version information
    - api-resources: List available API resources
    - cluster-info: Show cluster information
    - debug: debugs cluster resources using interactive containers
    - events: prints a table of the most important information about events.

    Blocked operations include: delete, update, patch, apply, create, replace, edit,
    scale, cordon, drain, taint, and any command with --overwrite flags.

    Args:
        command (str): The kubectl command to execute. Must start with "kubectl" and
                      be a read-only operation. Examples: "kubectl get pods",
                      "kubectl describe deployment nginx"

    Returns:
        str: The stdout output from the kubectl command, or an error message if the
             command fails, doesn't start with "kubectl", or contains disallowed operations.

    Example:
        Input: "kubectl get pods -n default"
        Returns: "NAME    READY   STATUS    RESTARTS   AGE\nnginx   1/1     Running   0          2d"
    """
    try:
        global current_kubectl_context

        # Check if command starts with kubectl
        if not command.startswith("kubectl "):
            return "Error: Command must start with 'kubectl'"

        # Extract the actual kubectl subcommand (after "kubectl ")
        kubectl_subcommand = command[8:]  # Remove "kubectl " prefix

        # List of allowed command prefixes (read-only operations)
        allowed_prefixes = [
            "get",
            "describe",
            "explain",
            "logs",
            "top",
            "config view",
            "config get-contexts",
            "debug",
            "version",
            "api-resources",
            "cluster-info",
            "events",
        ]

        # List of disallowed terms that might modify resources
        disallowed_terms = [
            "delete",
            "update",
            "patch",
            "apply",
            "create",
            "replace",
            "edit",
            "scale",
            "cordon",
            "drain",
            "taint",
            "label --overwrite",
            "annotate --overwrite",
        ]

        # Check if command is allowed
        is_allowed = any(
            kubectl_subcommand.startswith(prefix) for prefix in allowed_prefixes
        )
        has_disallowed = any(term in kubectl_subcommand for term in disallowed_terms)

        if not is_allowed or has_disallowed:
            return "Error: Only read-only kubectl commands are allowed (get, describe, etc.)"

        # Add context flag if a context has been switched and --context is not already in command
        if current_kubectl_context and "--context" not in command:
            command_parts = command.split()
            command_parts.insert(1, f"--context={current_kubectl_context}")
            command = " ".join(command_parts)
            logger.info(f"Modified command with context: {command}")
        else:
            logger.info(
                f"No context modification needed. Context: {current_kubectl_context}, has --context: {'--context' in command}"
            )

        # Execute the full command as provided
        logger.info(f"Executing command: {command}")
        result = subprocess.run(
            command.split(), capture_output=True, text=True, check=True
        )
        logger.info(
            f"Command successful, output length: {len(result.stdout)} characters"
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running kubectl command: {e.stderr}")
        return f"Error: {e.stderr}"
    except Exception as e:
        logger.error(f"Unexpected error in run_kubectl_command_ro: {e}")
        return f"Error: {e}"


@mcp.tool()
def run_argo_command(command: str):
    """
    Execute any argo command with full privileges (use with caution).

    This tool allows execution of any argo command, including potentially
    destructive operations like delete, stop, terminate, submit, etc. It provides
    complete access to your Argo server with the same permissions as
    your kubectl configuration.

    Args:
        command (str): The complete argo command to execute. Must start with "argo".
                      Examples: "argo submit -n argo --from clusterworkflowtemplate/automlai-dualsearch"

    Returns:
        str: The stdout output from the argo command, or an error message if the
             command fails or doesn't start with "argo".

    Example:
        Input: "argo -n argo list"
        Returns: "automl-dualsearch-pxd7s Succeeded 51d 49s 0"
    """
    try:
        global current_kubectl_context, current_namespace_context

        # Check if command starts with kubectl
        if not command.startswith("argo "):
            return "Error: Command must start with 'argo'"

        # Add context flag if a context has been switched and --context is not already in command
        if current_kubectl_context and "--context" not in command:
            command_parts = command.split()
            command_parts.insert(1, f"--context={current_kubectl_context}")
            command = " ".join(command_parts)
            logger.info(
                f"Using context: {current_kubectl_context} for command: {command}"
            )

        if current_namespace_context and ("--namespace" or "-n") not in command:
            command_parts = command.split()
            command_parts.insert(1, f"--namespace={current_namespace_context}")
            command = " ".join(command_parts)
            logger.info(
                f"Using context: {current_kubectl_context} and namespace: {current_namespace_context} for command: {command}"
            )

        # Execute the full command as provided
        result = subprocess.run(
            command.split(), capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running kubectl command: {e.stderr}")
        return f"Error: {e.stderr}"


# main entry point to run the MCP server
async def main():
    # Use run_async() in async contexts
    await mcp.run_async(transport="http", host="127.0.0.1", port=8000)


sc()
sn()

main_app = mcp.http_app()

# if __name__ == "__main__":
#    logger.info("Starting ArgoKubernetes MCP.")
#    mcp.run()
